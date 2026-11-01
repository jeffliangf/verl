"""Exercise all three policy turns with the actual frozen Qwen tokenizer, without paid tools."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from examples.video_agents import inspection_loop, inspection_rl
from examples.video_agents.storage import read_jsonl, write_json
from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
from verl.tools.tool_registry import initialize_tools_from_config
from verl.utils import hf_tokenizer
from verl.workers.rollout.replica import TokenOutput

root = Path(__file__).parent
model = Path('/home/jeff/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
args = inspection_rl.parse_args(['train', '--run-dir', str(root), '--data-dir', str(root.parent / 'data')])
config, _ = inspection_rl.training_config(args)
config.video_rl.output_dir = '/tmp/inspection-real-tokenizer-smoke'
tokenizer = hf_tokenizer(str(model), trust_remote_code=True)
row = read_jsonl(root.parent / 'data/train.jsonl')[0]
row['raw_prompt'] = row.pop('prompt')
texts = [
    'Reasoning before first generation.</think>\n<tool_call><function=sol_h3><parameter=prompt>A coherent scene with natural sound.</parameter><parameter=duration>5</parameter><parameter=seed>987</parameter></function></tool_call>',
    'Reasoning before inspection.</think>\n<tool_call><function=inspect_video><parameter=video_id>video_1</parameter></function></tool_call>',
    'Reasoning using the inspection.</think>\n<tool_call><function=sol_h3><parameter=prompt>A repaired scene preserving the original goal.</parameter></function></tool_call>',
]
requests, outputs = [], iter(texts)


async def server(**kwargs):
    requests.append(kwargs)
    ids = tokenizer.encode(next(outputs), add_special_tokens=False)
    return TokenOutput(token_ids=ids, log_probs=[-0.25] * len(ids), stop_reason='completed')


inspection_loop.generate_video = lambda directory, protocol, request: {
    'status': 'succeeded', 'request': request, 'prompt': request['prompt'],
    'path': str(directory / 'video.mp4'), 'sha256': 'mock',
}
inspection_loop.inspect_video = lambda *args: {
    'evidence': 'PRESERVED_INSPECTION_EVIDENCE', 'successful_elements_to_preserve': ['ORIGINAL_STRENGTH'],
}


async def main():
    loop = inspection_loop.InspectionLoop(
        trainer_config=DictConfigWrap(config), server_manager=SimpleNamespace(generate=server),
        tokenizer=tokenizer, processor=None, dataset_cls=None, data_config=DictConfigWrap(config.data),
        hf_model_type=json.loads((model / 'config.json').read_text())['model_type'],
        tools=ToolListWrap(initialize_tools_from_config(root / 'tools.json')),
    )
    output = await loop.run({}, **row)
    assert output.extra_fields['h3_result']['status'] == 'succeeded', output.extra_fields
    policy = [v for v, m in zip(output.response_ids, output.response_mask, strict=True) if m]
    expected = [v for text in texts for v in tokenizer.encode(text, add_special_tokens=False)]
    assert policy == expected
    last = tokenizer.decode(requests[-1]['prompt_ids'])
    assert 'Reasoning before first generation.' in last and 'PRESERVED_INSPECTION_EVIDENCE' in last
    result = {
        'status': 'passed', 'actual_tokenizer': type(tokenizer).__name__,
        'continuous_token_builder': type(loop.continuous_token_builder).__name__,
        'prompt_tokens': len(output.prompt_ids), 'policy_tokens': sum(output.response_mask),
        'observation_tokens': len(output.response_mask) - sum(output.response_mask),
        'turns': len(requests), 'final_video': output.extra_fields['h3_result']['request'],
    }
    write_json(root / 'native-tokenizer-smoke.json', result)
    print(json.dumps(result, indent=2))


asyncio.run(main())
