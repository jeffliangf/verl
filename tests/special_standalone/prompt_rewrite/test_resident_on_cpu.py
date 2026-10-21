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
"""Exercise cached Qwen generation on the same weights before and after training patches."""

import pytest

from examples.prompt_rewrite.config import RewriteConfig
from examples.prompt_rewrite.resident import capture_generation_forwards, generate_batch, generation_mode

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5")


def test_native_cached_generation_restores_training_forwards_and_weights(monkeypatch):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    from verl.models.transformers.monkey_patch import apply_monkey_patch

    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            patch_size=2,
        ),
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config)
    originals = capture_generation_forwards(config)
    assert Qwen3_5ForConditionalGeneration in originals
    # Register all global changes with pytest so other model tests see native classes.
    from transformers.integrations import flash_attention
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    other = capture_generation_forwards(modeling_qwen3_5_moe.Qwen3_5MoeConfig())
    for cls, forward in {**originals, **other}.items():
        monkeypatch.setattr(cls, "forward", forward)
    monkeypatch.setattr(flash_attention, "_flash_attention_forward", flash_attention._flash_attention_forward)
    for cls in (type(model.model.visual), modeling_qwen3_5_moe.Qwen3_5MoeVisionModel):
        monkeypatch.setattr(cls, "fast_pos_embed_interpolate", cls.fast_pos_embed_interpolate)

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False and kwargs["add_generation_prompt"] is True
            return [5, 6, 7]

        def decode(self, ids, **kwargs):
            return ",".join(map(str, ids))

    rewrite = RewriteConfig(max_new_tokens=5, max_model_len=32, backend="hf_fsdp2")
    conversations = [[{"role": "user", "content": "Original prompt"}]]
    pointers = [p.data_ptr() for p in model.parameters()]
    with generation_mode(model, originals):
        expected = generate_batch(model, Tokenizer(), conversations, rewrite, synced_gpus=False)
        assert not model.training and not torch.is_grad_enabled()
    assert model.training
    apply_monkey_patch(model, use_remove_padding=True, use_fused_kernels=False)
    training_forward = type(model).forward
    assert training_forward is not originals[type(model)]
    with generation_mode(model, originals):
        actual = generate_batch(model, Tokenizer(), conversations, rewrite, synced_gpus=False)
    assert actual == expected
    assert actual[0].output_tokens == 5 and actual[0].finish_reason == "length"
    assert pointers == [p.data_ptr() for p in model.parameters()]
    assert type(model).forward is training_forward and model.training
    # An exception must also restore training methods and mode.
    with pytest.raises(RuntimeError), generation_mode(model, originals):
        raise RuntimeError("generation failed")
    assert type(model).forward is training_forward and model.training
    assert all(p.grad is None for p in model.parameters())


def test_sft_reader_masks_the_exact_generation_prefix(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from examples.prompt_rewrite.config import build_messages
    from examples.prompt_rewrite.sft_dataset import PromptRewriteSFTDataset

    class CompleteConversationTokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            assert any(m["role"] == "user" for m in messages), "No user query found in messages"
            text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
            if add_generation_prompt:
                text += "<assistant>"
            return list(text.encode())

    messages = build_messages("An original prompt") + [{"role": "assistant", "content": "SECRET teacher target"}]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "train-1", "messages": messages, "enable_thinking": False}]), path)
    config = {"pad_mode": "no_padding", "truncation": "error", "max_length": 8192}
    data = PromptRewriteSFTDataset(path, CompleteConversationTokenizer(), config)
    item = data[0]
    prompt = bytes(item["input_ids"][item["loss_mask"] == 0].tolist()).decode()
    target = bytes(item["input_ids"][item["loss_mask"] == 1].tolist()).decode()
    assert prompt.endswith("<assistant>") and "SECRET" not in prompt
    assert target == "SECRET teacher target</assistant>"
    assert len(item["input_ids"]) < config["max_length"]
    data.max_length = 2
    with pytest.raises(ValueError, match="exceeds max_length"):
        data[0]
