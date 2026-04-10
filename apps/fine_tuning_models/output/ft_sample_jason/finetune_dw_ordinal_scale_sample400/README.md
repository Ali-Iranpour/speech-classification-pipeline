# DW Collusion Supervised Fine-Tuning Dataset

Dataset folder:
finetune_dw_ordinal_scale_sample400

## Purpose

Dataset for supervised fine-tuning of LLMs to predict perceived collusion
in political speech paragraphs.

## Input Data

Source file:
speech_sample_400_stratified.xlsx

Usable examples after cleaning:
400

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

Train: 320
Validation: 80

Stratified by score.

## Full Score Distribution

{1: 104, 2: 61, 3: 74, 4: 64, 5: 97}

## Train Score Distribution

{1: 83, 2: 49, 3: 59, 4: 51, 5: 78}

## Validation Score Distribution

{1: 21, 2: 12, 3: 15, 4: 13, 5: 19}