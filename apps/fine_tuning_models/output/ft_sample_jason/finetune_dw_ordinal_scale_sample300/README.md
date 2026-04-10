# DW Collusion Supervised Fine-Tuning Dataset

Dataset folder:
finetune_dw_ordinal_scale_sample300

## Purpose

Dataset for supervised fine-tuning of LLMs to predict perceived collusion
in political speech paragraphs.

## Input Data

Source file:
speech_sample_300_stratified.xlsx

Usable examples after cleaning:
300

Each paragraph has a survey variable:

share_yes ∈ [0,1]

representing the share of respondents labeling the paragraph as conspiratorial.

## Score Scaling

This dataset uses Ferenc's ordinal scaling:

0–19%   → 1
20–39%  → 2
40–59%  → 3
60–79%  → 4
80–100% → 5

## Training Target

The model learns to predict:

{"score": <int 1–5>}

given party affiliation and paragraph text.

## Dataset Files

OpenAI
- train_openai.jsonl
- val_openai.jsonl

Gemini
- train_gemini.jsonl
- val_gemini.jsonl

Open Models
- train_open.jsonl
- val_open.jsonl

## Split

Train: 240
Validation: 60

Stratified by score.

## Full Score Distribution

{1: 78, 2: 45, 3: 56, 4: 48, 5: 73}

## Train Score Distribution

{1: 62, 2: 36, 3: 45, 4: 38, 5: 59}

## Validation Score Distribution

{1: 16, 2: 9, 3: 11, 4: 10, 5: 14}