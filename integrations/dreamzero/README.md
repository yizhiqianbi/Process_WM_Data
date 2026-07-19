# DreamZero Integration

The data repository keeps DreamZero model changes as a small upstream patch. It does
not vendor model weights or the external repository.

Apply the patch to the pinned DreamZero revision before LoRA Pair inference:

```bash
cd /code/dreamzero
git apply --check /code/Process_WM_Data/integrations/dreamzero/dreamzero_lora_inference.patch
git apply /code/Process_WM_Data/integrations/dreamzero/dreamzero_lora_inference.patch
```

The patch fixes two inference contracts:

1. A `save_lora_only=true` checkpoint is loaded on top of the full DreamZero base
   checkpoint instead of attempting to fetch or load a second standalone Wan DiT.
2. LoRA inference honors model config overrides. Pair rollout uses this to enlarge
   the causal KV-cache from four training chunks to ten inference chunks without a
   cache reset midway through the 81-frame result.

`scripts/run_dreamzero_pair_inference.py` checks both patched arguments before it
loads the 16.5B model and fails early if the integration is missing.
