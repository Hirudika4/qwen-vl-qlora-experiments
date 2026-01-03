# Qwen-VL QLoRA Feasibility Experiments

This repository contains structured feasibility experiments for applying
QLoRA (Quantized Low-Rank Adaptation) to Qwen2.5-VL vision–language models.
The focus is on validating memory-efficient fine-tuning and inference setups
while preserving multimodal correctness and model stability.

## Objectives
- Validate 4-bit quantized loading of Qwen2.5-VL models
- Verify LoRA-based adaptation of the language backbone
- Ensure vision encoder and multimodal projector remain stable
- Confirm correct handling of multimodal inputs using image placeholders
- Evaluate Unsloth as a memory-optimized backend for QLoRA experiments

## Scope
The current scope focuses on feasibility and correctness rather than
full-scale fine-tuning or benchmark performance.

Included:
- Multimodal QLoRA smoke tests
- Forward pass and generation validation
- Real-image inference under quantization
- Unsloth-based memory-efficient model loading

Excluded (planned for later stages):
- Large-scale training runs
- Accuracy benchmarking across datasets
- Full 70B model fine-tuning

## Experiments
### Baseline QLoRA Validation
- Loads Qwen-VL with 4-bit quantization
- Applies LoRA adapters to the language backbone
- Verifies end-to-end multimodal inference

### Unsloth-Optimized QLoRA Validation
- Uses Unsloth for optimized 4-bit loading and memory efficiency
- Confirms LoRA injection does not disrupt multimodal routing
- Validates forward pass and generation using real images

## Notebooks
- `notebooks/qlora_qwen_vl_7B_smoke_test.ipynb`  
  Baseline QLoRA smoke test for Qwen-VL without Unsloth.

- `notebooks/qwen_vl_qlora_unsloth_smoke_test.ipynb`  
  QLoRA smoke test using Unsloth, validating memory-efficient multimodal
  inference and generation.

## Environment
- Frameworks: PyTorch, Hugging Face Transformers
- Quantization: bitsandbytes (4-bit)
- Optimization: LoRA / QLoRA
- Hardware: Google Colab (T4 GPU) for smoke tests

## Status
This repository currently demonstrates feasibility and technical validation.
The next phase will focus on memory benchmarking, controlled comparisons
(QLoRA vs FP16), and scaling analysis toward larger model variants.

