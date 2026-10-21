# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate rewrites with the resident FSDP2 training model and its tokenizer."""

from contextlib import contextmanager
from datetime import timedelta
from importlib import import_module
from pathlib import Path

from .config import RewriteConfig, build_messages
from .evaluation import EvaluationRun
from .reporting import print_report
from .rewrite import RewriteResult, export_rewrites, load_rewrites, save_rewrite
from .storage import freeze_json, run_lock


def capture_generation_forwards(hf_config) -> dict:
    """Capture native HF forwards before the engine installs training-only patches.

    In particular, verl's Qwen linear-attention training patch expects a legacy
    cache interface. Native HF generation must use the matching HF cache/forward
    implementation; swapping methods does not copy or reload any weights.
    """
    import torch
    from transformers.models.auto.auto_factory import _get_model_class

    from verl.utils.model import get_hf_auto_model_class

    # AutoModel classes are factories, not the module classes whose forwards the
    # engine patches. Resolve the concrete class without allocating any weights.
    auto_class = get_hf_auto_model_class(hf_config)
    model_class = _get_model_class(hf_config, auto_class._model_mapping)
    module = import_module(model_class.__module__)
    forwards = {
        cls: cls.forward
        for cls in vars(module).values()
        if isinstance(cls, type)
        and issubclass(cls, torch.nn.Module)
        and cls.__module__ == module.__name__
        and "forward" in vars(cls)
    }
    if model_class not in forwards:
        raise ValueError(f"Cannot capture the native forward for {model_class.__name__}")
    return forwards


@contextmanager
def generation_mode(model, native_forwards: dict):
    """Disable training behavior/gradients, then restore every training forward."""
    import torch

    training_forwards = {cls: cls.forward for cls in native_forwards}
    was_training = model.training
    try:
        for cls, forward in native_forwards.items():
            cls.forward = forward
        model.eval()
        # no_grad is compatible with reusable FSDP communication buffers.
        with torch.no_grad():
            yield
    finally:
        for cls, forward in training_forwards.items():
            cls.forward = forward
        model.train(was_training)


def generate_batch(model, tokenizer, conversations: list[list[dict]], config: RewriteConfig, *, synced_gpus: bool):
    """Greedily decode assistant continuations, preserving EOS/truncation status."""
    import torch
    from transformers import GenerationConfig

    tokens = [
        tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
        )
        for messages in conversations
    ]
    width = max(map(len, tokens))
    if width + config.max_new_tokens > config.max_model_len:
        raise ValueError("Prompt and output budget exceed max_model_len")
    eos = model.generation_config.eos_token_id
    eos = eos if eos is not None else tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else eos
    if not eos_ids:
        raise ValueError("Generation requires an EOS token")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    device = next(model.parameters()).device
    input_ids = torch.full((len(tokens), width), pad, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(tokens):
        input_ids[index, -len(sequence) :] = torch.tensor(sequence, device=device)
        attention_mask[index, -len(sequence) :] = 1
    generation = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=config.max_new_tokens,
        eos_token_id=eos_ids,
        pad_token_id=pad,
        use_cache=True,
        repetition_penalty=1.0,
        return_dict_in_generate=False,
    )
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation,
        synced_gpus=synced_gpus,
    )
    results = []
    for sequence in output[:, width:].tolist():
        end = next((i for i, token in enumerate(sequence) if token in eos_ids), None)
        generated = sequence if end is None else sequence[: end + 1]
        results.append(
            RewriteResult(
                tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                "length" if end is None else "stop",
                len(generated),
            )
        )
    return results


class ResidentEvaluator:
    """All ranks decode; rank zero persists outputs, calls H3/Gemini, and reports."""

    def __init__(self, trainer, native_forwards: dict):
        import torch.distributed as dist
        from torch.distributed.fsdp import register_fsdp_forward_method

        self.trainer, self.native_forwards = trainer, native_forwards
        self.settings = trainer.config.video_eval
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        # Remote video generation/judging can take hours. Wait on CPU collectives,
        # with no outstanding CUDA collective and no second GPU model.
        self.control = dist.new_group(backend="gloo", timeout=timedelta(hours=24))
        register_fsdp_forward_method(trainer.engine.module, "generate")

    def on_rank_zero(self, fn):
        """Broadcast root-side results and failures so all ranks follow the same stages."""
        import torch.distributed as dist

        message = [None]
        if self.rank == 0:
            try:
                message[0] = {"value": fn()}
            except Exception as exc:
                message[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(message, src=0, group=self.control)
        if "error" in message[0]:
            raise RuntimeError(f"Epoch video evaluation stopped: {message[0]['error']}")
        return message[0]["value"]

    def evaluate(self, step: int):
        """Evaluate this epoch, resuming completed rewrites/videos/judgments in place."""
        import torch
        import torch.distributed as dist

        trainer, settings = self.trainer, self.settings
        path = Path(settings.run_dir).expanduser().resolve() / "eval" / f"global_step_{step}"
        config = RewriteConfig(
            model=(
                str(trainer.config.model.path)
                if step == 0
                else str(Path(trainer.config.trainer.default_local_dir).resolve() / f"global_step_{step}/huggingface")
            ),
            backend="hf_fsdp2",
            max_new_tokens=settings.max_new_tokens,
            max_model_len=settings.max_model_len,
        )

        def prepare():
            print(f"Video evaluation at step {step}: preparing {path}", flush=True)
            run = EvaluationRun.prepare(
                path,
                name=f"{path.parent.parent.name}-step-{step}",
                config=config,
                dataset=Path(settings.dataset),
                references=Path(settings.references),
                eval_repo=Path(settings.eval_repo),
            )
            with run_lock(path):
                freeze_json(path / "rewrite-source.json", {"mode": "policy", "rewriter": run.protocol["rewriter"]})
                completed = load_rewrites(path, run.protocol, require_complete=False)
            pending = [sample for sample in run.protocol["samples"] if sample["id"] not in completed]
            # Fail on any overlong prompt before ranks enter FSDP collectives.
            for sample in pending:
                tokens = trainer.model_config.tokenizer.apply_chat_template(
                    build_messages(sample["prompt"]),
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if len(tokens) + config.max_new_tokens > config.max_model_len:
                    raise ValueError(f"Eval prompt exceeds context budget: {sample['id']}")
            return pending

        pending = self.on_rank_zero(prepare)
        model, tokenizer = trainer.engine.module, trainer.model_config.tokenizer
        if pending:
            device = next(model.parameters()).device
            devices = [device.index] if device.type == "cuda" else []
            with (
                torch.random.fork_rng(devices=devices),
                trainer.engine.eval_mode(),
                generation_mode(model, self.native_forwards),
            ):
                for offset in range(0, len(pending), settings.batch_size * self.world):
                    start = offset + self.rank * settings.batch_size
                    batch = pending[start : start + settings.batch_size]
                    # Every rank must participate, including the last partially filled round.
                    inputs = batch or [pending[-1]]
                    results = generate_batch(
                        model,
                        tokenizer,
                        [build_messages(s["prompt"]) for s in inputs],
                        config,
                        synced_gpus=self.world > 1,
                    )
                    local = list(zip(batch, results[: len(batch)], strict=True))
                    gathered = [None] * self.world
                    dist.all_gather_object(gathered, local, group=self.control)

                    def persist(entries_by_rank=gathered, batch_offset=offset):
                        run = EvaluationRun(path)
                        with run_lock(path):
                            for entries in entries_by_rank:
                                for sample, result in entries:
                                    save_rewrite(path, run.protocol, sample, result)
                        completed = min(batch_offset + settings.batch_size * self.world, len(pending))
                        print(f"Video evaluation at step {step}: rewrites {completed}/{len(pending)}", flush=True)

                    self.on_rank_zero(persist)

        def finish():
            run = EvaluationRun(path)
            with run_lock(path):
                export_rewrites(path, run.protocol)
            status = run.generate(execute=True, concurrency=settings.h3_concurrency, neutral_on_failure=True)
            if status["videos_accounted_for"] != status["prompts"]:
                raise RuntimeError("Candidate videos are incomplete; inspect state files and resume this run")
            run.judge(execute=True, concurrency=settings.judge_concurrency, neutral_on_failure=True)
            report = run.report(step=step)
            print_report(report)

        self.on_rank_zero(finish)
