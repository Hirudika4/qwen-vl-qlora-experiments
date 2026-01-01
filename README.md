# Qwen-VL QLoRA Feasibility Experiments

This repository contains initial feasibility experiments for applying
QLoRA (Quantized Low-Rank Adaptation) to Qwen2.5-VL models.

## Objective
The goal of this work is to verify whether QLoRA can be applied to
large vision–language models without breaking multimodal processing.

## What is Verified
- 4-bit quantized loading of Qwen2.5-VL
- Successful LoRA adapter injection into the language backbone
- Correct multimodal input construction using image placeholders
- End-to-end forward pass and text generation

## Notebook
- `notebooks/qlora_qwen_vl_smoke_test.ipynb`  
  Contains a minimal smoke test validating QLoRA compatibility.

