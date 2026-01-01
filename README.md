# Qwen-VL QLoRA Experiments

This repository contains practical feasibility experiments for applying QLoRA
(Quantized Low-Rank Adaptation) to Qwen2.5-VL vision–language models.

## Overview
The goal is to validate that QLoRA can be used with Qwen-VL models while preserving
multimodal functionality and inference stability.

## Scope
- 4-bit quantized model loading
- LoRA-based parameter-efficient adaptation of the language backbone
- Multimodal input processing with image placeholders
- End-to-end multimodal inference and generation

## Artifacts
- `notebooks/qlora_qwen_vl_smoke_test.ipynb` — QLoRA compatibility smoke test
