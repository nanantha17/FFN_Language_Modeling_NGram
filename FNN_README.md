# FNN N-gram Language Model
## Feedforward Neural Network for Language Modeling — Built from Scratch in PyTorch

Deliberate ML foundations project: built a feedforward neural network language model
from scratch to understand failure modes at the architecture level, not just the program level.
Compared 2-gram, 4-gram, and 8-gram context windows. Tracked all experiments in W&B.

**W&B Report:**
https://wandb.ai/nishanth-anantha87-asml/ngram-song-generator/reports/FFN-for-Language-Model-N-gram-model-comparison-Song-generator---VmlldzoxNjUwMzA4NA

---

## What This Project Is

Not a tutorial completion. A deliberate experiment to answer one question:
*where exactly does a fixed-context architecture break, and why does attention solve it?*

Built on a Rick Astley corpus (~150 tokens) — intentionally small to make failure modes
visible and measurable rather than hidden in noise.

---

## Stack

| Component | Technology |
|---|---|
| Framework | PyTorch |
| Tokenization | NLTK word_tokenize |
| Embeddings | Custom embedding layer (vocab_size × dim=10) |
| Architecture | FNN: embedding → Linear+ReLU (128 hidden) → output logits |
| Training | CrossEntropyLoss → backprop → SGD → StepLR |
| Context sizes | N = 2, 4, 8 |
| Experiment tracking | W&B (perplexity, loss, generated samples per run) |
| Visualization | t-SNE 2D projection of learned word vectors |

---

## Architecture

```
Raw text (Rick Astley corpus, ~150 tokens)
    ↓
word_tokenize → lowercase → remove punctuation
    ↓
Counter → Vocab → stoi/itos mappings
    ↓
N-gram pair generation: [context words] → target word
N = 2, 4, 8 (three parallel experiments)
    ↓
DataLoader + padding (batch_size=10, collate_batch)
    ↓
NgramLanguageModeler:
  Embedding layer (vocab_size × embedding_dim=10)
      ↓ reshape
  Linear + ReLU (N×10 → 128 hidden units)
      ↓
  Output layer (128 → vocab_size logits)
    ↓
Training loop: CrossEntropyLoss → backprop → SGD → StepLR
    ↓
  Loss + perplexity (np.exp(loss) per epoch)
  t-SNE embedding visualization
  Text generation: write_song() → argmax / temperature sampling
```

---

## Key Findings

### Finding 1 — Larger context window reduces loss, but on a small corpus this is memorization

The 8-gram model achieves near-zero loss and perplexity ~1.0. This looks like success —
it is actually the model memorizing the 150-token corpus.

With a context window of 8 words and a vocabulary of ~50 unique words, the model has
seen nearly every possible sequence during training. There is no held-out distribution to generalize to.

| Model | Final Loss | Final Perplexity | Interpretation |
|---|---|---|---|
| 2-gram | ~1.8 | ~6.0 | Limited context, genuine uncertainty |
| 4-gram | ~0.9 | ~2.5 | More context, some memorization |
| 8-gram | ~0.05 | ~1.05 | Near-total memorization |

**Insight:** On a real dataset (10M+ tokens), the 2-gram vs 8-gram difference would reflect
genuine language understanding. Corpus size matters as much as architecture.

---

### Finding 2 — The coherence wall: why the model fails after ~5 words

All three models produce repetitive loops ("never gonna never gonna never gonna...")
within 5-8 words of generation. This is the fixed context window problem in action.

The model can only reference the previous N tokens. Once it generates a repeated sequence,
it has no mechanism to "remember" that it already produced that phrase.
There is no long-range dependency.

**This is exactly the problem attention mechanisms solve.**

A transformer allows any token to attend to any previous token regardless of distance.
A transformer trained on this corpus would learn that "never gonna" almost always
precedes one of five specific phrases — not because of immediate context but because
of global pattern recognition across the full sequence.

---

### Finding 3 — Embedding visualization shows emergent semantic structure

The t-SNE plots show learned word relationships from co-occurrence patterns alone,
with no explicit semantic supervision:

- Action words ("give," "run," "tell," "make") cluster together
- Emotional words ("hurt," "cry") form a separate cluster
- Function words distribute across the space

This emerges purely from predicting the next word — the same principle underlying
word2vec and the embedding layers in every modern LLM, scaled to billions of tokens.

---

### Finding 4 — What would actually fix this (in order of impact)

| Fix | Expected Impact |
|---|---|
| Larger corpus (full album, Kaggle genre dataset) | Removes memorization; makes N-gram comparison meaningful |
| Validation split | Exposes true generalization vs training perplexity |
| Beam search instead of argmax | Reduces repetitive loops; argmax always picks single most likely word |
| Single attention head | Would outperform 8-gram FNN on any corpus larger than ~500 tokens |

---

## W&B Experiment Tracking

Three runs logged in parallel — 2-gram, 4-gram, 8-gram — each tracking:
- Loss curve per epoch (100 epochs)
- Perplexity per epoch
- Generated text sample at epoch 100

The W&B dashboard makes the architectural comparison visible:
8-gram drives perplexity to ~1 fastest — looks best on the chart,
is worst in terms of generalization.

**This is a program insight, not just an ML insight:**
metrics that look like success can mask the wrong behavior.
Perplexity on training data is not the same as perplexity on held-out data.
The eval definition matters as much as the number.

---

## What's Next

1. Rebuild with a single attention head on the same corpus — measure the delta directly
2. Add validation split to expose true generalization gap
3. Train on a larger corpus (Kaggle lyrics dataset) to make N-gram comparison meaningful
4. Continue alongside W&B MLOps for deeper understanding of failure modes at architecture level

---

## Running the Model

```bash
pip install torch wandb nltk scikit-learn matplotlib

python ngram_language_model.py --context 2 --epochs 100
python ngram_language_model.py --context 4 --epochs 100
python ngram_language_model.py --context 8 --epochs 100
```

W&B logs automatically to the `ngram-song-generator` project.

---

## Certifications

Built as part of:
- Deep Learning with TensorFlow — Kaggle (Apr 2026)
- W&B LLM Evaluation (2026)
- IBM GenAI & LLMs (Feb 2026)
