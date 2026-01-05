# Qwen-VL QLoRA Feasibility Experiments

This repository contains feasibility and validation experiments for applying
QLoRA (Quantized Low-Rank Adaptation) to **Qwen2.5-VL** vision–language models.
The work focuses on validating memory-efficient adaptation and inference
pipelines while preserving correct multimodal behavior.

## Objectives
- Validate 4-bit quantized loading of Qwen2.5-VL models
- Verify LoRA-based adaptation of the language backbone
- Ensure vision encoder and multimodal projector remain stable
- Confirm correct handling of multimodal inputs using image placeholders
- Evaluate Unsloth as a memory-optimized backend for QLoRA experiments
- Analyze memory sensitivity of key hyperparameters in multimodal QLoRA setups

## Scope
The current scope focuses on feasibility, correctness, and memory behavior
rather than full-scale fine-tuning or benchmark performance.

Included:
- Multimodal QLoRA smoke tests
- Forward pass and generation validation
- Real-image inference under quantization
- Unsloth-based memory-efficient model loading
- Batch size sensitivity analysis for multimodal inputs

## Experiments

### Baseline QLoRA Validation
- Loads Qwen-VL with 4-bit quantization (NF4)
- Applies LoRA adapters to the language backbone only
- Verifies end-to-end multimodal inference and generation
- Confirms vision encoder and multimodal projector remain frozen

### Unsloth-Optimized QLoRA Validation
- Uses Unsloth for optimized 4-bit model loading
- Enables Unsloth gradient checkpointing for memory efficiency
- Confirms LoRA injection does not disrupt multimodal routing
- Validates forward pass and generation using real images

### Experiment D — Batch Size Sensitivity (Memory Feasibility)
- Evaluates how batch size impacts VRAM usage in a multimodal QLoRA setup
- Configuration:
  - Model: Qwen2.5-VL-7B
  - Quantization: 4-bit NF4
  - LoRA: r=64, α=16 (language backbone only)
  - Vision encoder: frozen
- Runs:
  - Batch size = 1
  - Batch size = 2
- Measurements:
  - Peak VRAM usage (`torch.cuda.max_memory_allocated`)
  - Runtime stability (OOM vs successful execution)
- Key finding:
  - Increasing batch size results in a measurable VRAM increase due to image
    activation memory, confirming batch size as a sensitive hyperparameter
    in vision–language QLoRA training.

## Notebooks
- `notebooks/qlora_qwen_vl_7B_smoke_test.ipynb`  
  Baseline QLoRA smoke test for Qwen-VL.

- `notebooks/qlora_qwen_vl_7B_unsloth_smoke_test.ipynb`  
  Unsloth-optimized QLoRA smoke test validating memory-efficient multimodal
  inference and generation.

- `notebooks/qwen_vl_qlora_unsloth_batchsize_memory_test_ipynb.ipynb`  
  Controlled experiment measuring VRAM usage and stability across batch sizes
  in a multimodal QLoRA setup.

## Environment
- Frameworks: PyTorch, Hugging Face Transformers
- Quantization: bitsandbytes (4-bit NF4)
- Optimization: LoRA / QLoRA / Unsloth
- Hardware: Google Colab (T4 GPU) for feasibility and memory experiments

## Status
This repository demonstrates successful feasibility validation of QLoRA
with Qwen-VL, including multimodal correctness, Unsloth compatibility, and
memory sensitivity analysis. The results provide a stable foundation for
subsequent scaling and benchmarking work.
