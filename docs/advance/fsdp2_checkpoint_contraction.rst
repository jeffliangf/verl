Contract a native FSDP2 checkpoint
=================================

Last updated: 11/09/2026

The native per-rank checkpoint loader uses the current world size in its model,
optimizer and extra-state filenames. A checkpoint saved with sixteen ranks
cannot be loaded with eight ranks by changing the launch configuration alone.

For the supported layout, an offline conversion retains the training state::

   python -m verl.utils.checkpoint.reshard_fsdp2 \
     --source /source/checkpoints/global_step_80 \
     --destination /converted/checkpoints/global_step_80 \
     --world-size 8

This tool requires the validated PyTorch 2.13 serialization format and a native
FSDP2 checkpoint with a completion marker, dataloader state and Hugging Face
metadata. It supports an integer contraction of an evenly sharded, one-dimensional
full-world ``Shard(0)`` mesh named ``fsdp``. Uneven tensors, other meshes, arbitrary
replicated tensors and pending asynchronous queues are rejected. Existing output
directories are never overwritten. An interrupted conversion remains in a hidden
``.global_step_80.resharding`` directory and is not a usable checkpoint.

The tool joins neighboring local shards without changing parameter or Adam
moment precision. It retains scalar optimizer steps, parameter groups, scheduler,
model state-dict metadata, and the unchanged dataloader file. Each new rank uses
the RNG state of the first old rank contributing to its shard. A changed world
size does not promise a bit-identical subsequent training trajectory.

Source files are memory-mapped; only one destination rank's model or optimizer
state is materialized at a time. Plan CPU RAM for that destination shard, the
readback verification and filesystem cache. Every output tensor is checked
against the in-memory conversion, and every resulting file gets a SHA-256 digest
in ``reshard-verification.json``. The output directory is published only after
all ranks and ancillary state have been written and verified. The source remains
unchanged.

Native CUDA-mesh metadata is retained while conversion runs entirely on CPU.
This uses PyTorch's detached DTensor serialization internals, not a new training
checkpoint format. Validate a native GPU load, optimizer update, save and restart
on the destination before relying on the converted checkpoint for a long run.
