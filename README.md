# Qwen-VL QLoRA Feasibility Experiments

This repository contains feasibility and validation experiments for applying
QLoRA (Quantized Low-Rank Adaptation) to **Qwen2.5-VL** vision–language models.
The work focuses on validating memory-efficient adaptation and inference
pipelines while preserving correct multimodal behavior.

## Purpose
Large vision–language models provide strong multimodal reasoning capabilities
but are challenging to adapt under limited GPU memory. QLoRA enables efficient
fine-tuning through 4-bit quantization and low-rank updates.

The goal of this project is to determine whether QLoRA can be safely applied to
Qwen-VL models without breaking:
- image–text alignment
- multimodal routing
- inference stability and generation behavior

## What This Repository Validates
- 4-bit quantized loading of Qwen2.5-VL models
- LoRA-based parameter-efficient adaptation of the language backbone
- Stability of the vision encoder and multimodal projector
- Correct handling of multimodal inputs using image placeholder tokens
- End-to-end multimodal forward pass and text generation
- Compatibility of **Unsloth** with QLoRA for memory-optimized execution

## Experiments
### Baseline QLoRA Smoke Test
- Loads Qwen-VL with 4-bit quantization
- Injects LoRA adapters into the language backbone
- Verifies end-to-end multimodal inference

### Unsloth-Optimized QLoRA Smoke Test
- Uses Unsloth for optimized 4-bit loading and reduced memory footprint
- Confirms LoRA injection does not disrupt multimodal processing
- Validates inference and generation using real images

## Notebooks
- `notebooks/qlora_qwen_vl_7B_smoke_test.ipynb`  
  Baseline QLoRA smoke test for Qwen-VL.

- `notebooks/qlora_qwen_vl_7B_unsloth_smoke_test.ipynb`  
  Unsloth-optimized QLoRA smoke test validating memory-efficient multimodal
  inference and generation.

## Environment
- Frameworks: PyTorch, Hugging Face Transformers
- Quantization: bitsandbytes (4-bit)
- Optimization: LoRA / QLoRA
- Hardware: Google Colab (T4 GPU) for feasibility testing

## Status
This repository currently serves as a technical validation and feasibility
baseline. Results from these experiments will inform subsequent scaling and
benchmarking efforts.
