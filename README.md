# In-House Legal Intelligence System

A compact, self-hosted legal NLP pipeline developed for the **CS781 Capstone Project** at the Indian Institute of Technology Kanpur. The project studies training small Transformer language models from scratch on Indian legal text and adapting them to six downstream legal NLP tasks under limited compute and data constraints.

## Project Overview

The system follows seven phases of the IL-TUR-style benchmark:

1. Legal language-model pretraining
2. Legal named entity recognition (NER)
3. Rhetorical role prediction
4. Court judgment prediction
5. Legal statute identification
6. Abstractive judgment summarisation
7. Prior case retrieval (PCR)

All Phase-1 backbones were trained from random initialisation. No pretrained model weights, pretrained tokenizers, external legal corpora, or LLM APIs were used for model training. The downstream phases reuse the Phase-1 checkpoints and compare parameter-efficient fine-tuning (PEFT), partial/full tuning, and knowledge distillation.

## Best Official Results

Best valid Codabench results recorded in the report; leaderboard snapshot: **9 November 2026**.

| Phase | Task | Best primary metric |
|---:|---|---:|
| 1 | Legal LM Pretraining | BERTScore F1: **0.8251** |
| 2 | Legal NER | Strict macro-F1: **0.5535** |
| 3 | Rhetorical role prediction | Macro-F1: **0.4695** |
| 4 | Court judgment prediction | Macro-F1: **0.6730** |
| 5 | Legal statute identification | Macro-F1: **0.3342** |
| 6 | Abstractive summarisation | BERTScore F1: **0.8447** |
| 7 | Prior case retrieval | Micro-F1@7: **0.4306** |

### Best single experiment vs. best ensemble

| Phase | Metric | Best single experiment | Best ensemble |
|---:|---|---:|---:|
| 2 – NER | Strict macro-F1 | 0.5400 | **0.5535** |
| 3 – Rhetorical roles | Macro-F1 | **0.4695** | 0.4685 |
| 4 – Judgment | Macro-F1 | **0.6730** | 0.6730 |
| 5 – Statutes | Macro-F1 | 0.3303 | **0.3342** |
| 6 – Summarisation | BERTScore F1 | **0.8447** | 0.7997 |
| 7 – Retrieval | Micro-F1@7 | **0.4306** | 0.4288 |

## Phase-1 Foundation Models

Five hand-written PyTorch Transformer variants were implemented with tied input/output embeddings. All were trained from random initialisation and had approximately 94–110 million parameters.

| Model | Architecture | Main design choices | Parameters | Test BERTScore F1 |
|---|---|---|---:|---:|
| A1 | Decoder-only | 12 pre-norm layers, LayerNorm, GELU MLP, sinusoidal positions | 94.2M | 0.8176 |
| A2 | Encoder–decoder | 6 encoder + 6 decoder layers, cross-attention | 108.4M | 0.7838 |
| A3 | Hybrid prefix-LM | 14-layer stack; bidirectional prompt and causal continuation | 108.4M | 0.7536 |
| A4 | Lightweight decoder | RoPE, RMSNorm, SwiGLU, grouped-query attention (GQA) | 109.9M | **0.8251** |
| A5 | RoPE + QK-Norm | 14 layers, RMSNorm, SwiGLU, full multi-head attention and per-head QK normalisation | 108.3M | Not submitted |

**Phase-1 leaderboard model: A4.** It combines rotary positional embeddings, RMSNorm, SwiGLU feed-forward layers and grouped-query attention.

A5 was not submitted to Phase 1, but it produced the best single experiment in every downstream phase (2–7) in the reported experiments.

### Tokenizer and pretraining setup

- Custom byte-level BPE tokenizer implemented from scratch.
- Vocabulary size: **12,000**, including 256 byte values and four special tokens.
- Merge learning used 8M characters, restricted to the 100,000 most frequent distinct words.
- Phase-1 training data: a streamed **300M-token** slice of the Indian legal corpus.
- Validation data: **10M tokens**.
- Context/block length: **512 tokens**.
- Optimizer: AdamW; peak learning rate `3e-4`; 200 warm-up steps followed by cosine decay.
- Weight decay: `0.1`; dropout: `0.1`; gradient clipping: `1.0`; mixed precision: FP16.
- Training batch: 16 sequences × 4 gradient-accumulation steps.

## Downstream Adaptation Methods

Each task notebook loads a Phase-1 backbone, replaces the language-model head with a task-specific head, and applies the selected adaptation method. The implemented methods include:

- Full fine-tuning and partial tuning (last four blocks)
- LoRA and rank-stabilised LoRA (rsLoRA)
- LoRMA (multiplicative low-rank adaptation)
- IA³
- Houlsby and Pfeiffer adapters
- BitFit
- QLoRA with block-wise 4-bit NF4 frozen weights
- Knowledge distillation (KD) with a reduced-depth student

The adaptation methods were hand-written. Identity-at-initialisation checks were run for the LoRA family, IA³ and adapters across all five backbones.

### Task-specific pipelines

| Phase | Task | Main approach |
|---:|---|---|
| 2 | Legal NER | Token classification with 25 BIO tags, overlapping 512-token windows, centre-weighted log-probability aggregation and BIO-constrained Viterbi decoding on uncertain stretches. |
| 3 | Rhetorical role prediction | Contextual sentence representations from overlapping windows, mean pooling, relative sentence position and a per-sentence softmax classifier. CL and IT are trained jointly. |
| 4 | Court judgment prediction | Long judgments split into 218-word chunks; the last four chunks are encoded, mean-pooled and combined with learned chunk attention and a small MLP. |
| 5 | Statute identification | Independent sigmoid output for each of 100 statute IDs, weighted binary cross-entropy and a dev-tuned global threshold. |
| 6 | Judgment summarisation | Importance-based sentence selection to fit the 512-token input, followed by generation. Boost variants use multiple passes, trigram de-duplication and extractive top-up. |
| 7 | Prior case retrieval | Shared bi-encoder with mean-pooled chunks, 256-dimensional projection, L2 normalisation and contrastive InfoNCE training. Dense scores are fused with TF-IDF. |

## Key Findings

- **Phase 1:** A4 achieved the best submitted score (0.8251). A2 and A3, which apply loss to only part of each sequence, scored lower. Sparse supervision is a possible explanation, not a proven cause; this was not a controlled ablation.
- **Downstream transfer:** A5 gave the best single experiment in every downstream phase, despite not being submitted to Phase 1. Its advantage over A4 has not been isolated because the models differ in QK-Norm, attention type and depth.
- **Phase 2:** The best ensemble improved strict macro-F1 from 0.5400 to 0.5535.
- **Phase 5:** The best ensemble reached 0.3342 macro-F1. Errors were concentrated among frequent neighbouring statutes rather than a long tail of very rare labels.
- **Phase 6:** Model-only summaries were short (about 22–35 words for A5 runs). The strongest leaderboard results relied substantially on extractive filling; they should not be interpreted as evidence of strong abstractive generation.
- **Phase 7:** TF-IDF was much stronger than the dense retrievers on development data. The fused submissions were therefore largely driven by lexical retrieval.

Most comparisons use a single seed. Small score differences should be interpreted cautiously.

## Data and Compute

- **Dataset:** `Exploration-Lab/CS-781-Capstone` and the official phase configurations.
- **Phase-1 legal sources:** Acts, High Courts, Supreme Court and Tribunals.
- **Corpus size:** More than 4.5B tokens across the four sources; a 300M-token training slice and 10M validation tokens were used.
- **External data:** No external legal corpus was added.
- **Compute:** Kaggle T4 GPUs were the main training platform, with session limits and checkpoint/resume logic.
- **Long-document constraint:** The context was 512 tokens, so downstream tasks used chunking, selection or aggregation strategies.

## Limitations

- Models were trained from scratch on a subset of the legal corpus and are not expected to match pretrained models on tasks requiring broad language knowledge.
- The 512-token context requires truncation, chunk selection or other long-document handling.
- Most comparisons use a single random seed; differences around 0.01 may be within run-to-run variation.
- There was no human evaluation by legal professionals.
- Phase-6 scores are substantially influenced by extractive filling.
- Phase-7 dense retrieval underperformed TF-IDF on development data.

## References

1. IL-TUR: *Benchmark for Indian Legal Text Understanding and Reasoning*. arXiv:2407.05399, 2024.
2. Bihany, H., Patel, S., and Modi, A. *LoRMA: Low-Rank Multiplicative Adaptation for LLMs*. arXiv:2506.07621, 2025.
3. Dong, L. et al. *Unified Language Model Pre-training for Natural Language Understanding and Generation*. NeurIPS, 2019.
4. Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
5. Zhang, B. and Sennrich, R. *Root Mean Square Layer Normalization*. NeurIPS, 2019.
6. Shazeer, N. *GLU Variants Improve Transformer*. arXiv:2002.05202, 2020.
7. Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP, 2023.
8. Henry, A. et al. *Query-Key Normalization for Transformers*. Findings of EMNLP, 2020.
9. Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021.
10. Kalajdzievski, D. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*. arXiv:2312.03732, 2023.
11. Liu, H. et al. *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning*. arXiv:2205.05638, 2022.
12. Houlsby, N. et al. *Parameter-Efficient Transfer Learning for NLP*. arXiv:1902.00751, 2019.
13. Pfeiffer, J. et al. *AdapterFusion: Non-Destructive Task Composition for Transfer Learning*. EACL, 2021.
14. Ben Zaken, E., Goldberg, Y., and Ravfogel, S. *BitFit: Simple Parameter-Efficient Fine-Tuning for Transformer-Based Masked Language Models*. ACL, 2022.
15. Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314, 2023.
16. Hinton, G., Vinyals, O., and Dean, J. *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531, 2015.

## Citation

```text
Aditya Kumar. In-House Legal Intelligence System.
CS781 Capstone Project, Indian Institute of Technology Kanpur, 2026.
```
