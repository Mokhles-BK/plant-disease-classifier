Always respond in English only. No Chinese.
# Plant Disease Classifier — Project Context

## Goal
End-to-end deployed web app: upload leaf photo → get disease diagnosis +
confidence score. Not a notebook. Must ship as a public URL, free tier.

## Stack decisions (don't relitigate these)
- PyTorch, CPU-only (no GPU available).
- Dataset: `GVJahnavi/PlantVillage_dataset` on HF Hub — 38 classes,
  43,503 train / 10,878 test images, 256x256 RGB. Loads via plain
  `datasets.load_dataset()`, no auth needed.
- DO NOT use `mohanty/PlantVillage` — broken (Python 2.7 loader script,
  401 on data.zip). Already tried, confirmed dead. Don't re-suggest it.
- Deployment target: Gradio app on Hugging Face Spaces. NOT Docker,
  NOT FastAPI, NOT Render/Railway for v1. Decided to avoid container
  complexity for the first shipped version — may revisit as "v2 REST
  API" later but not now.

## Project structure / status
- `src/config.py` — done. Don't touch without asking first.
- `src/utils.py` — done. Has `set_seed`, `get_device`,
  `compute_class_weights` (returns weights + optional
  `WeightedRandomSampler`), `save_json`/`load_json`, `topk_from_logits`.
- `src/data.py` — done. `make_loaders(max_samples=None)` returns
  train/val/test loaders + class_weights + sampler. `--max-samples N`
  flag caps each class to N rows for fast dev smoke-tests.
- `src/models.py` — done. `build_model(name)` where name is
  `"baseline"` (small CNN from scratch, no pretrained weights) or
  `"transfer"` (resnet18, frozen backbone + new head). NOTE:
  `freeze_backbone` logic was buggy once (requires_grad inverted,
  backbone was training unfrozen) — already fixed. Don't reintroduce
  that bug.
- `src/train.py` — done. `train(model_name, epochs, max_samples, ...)`.
  Checkpoints on best val_loss (not last epoch), early stopping, saves
  a run-history JSON next to the checkpoint. Loss uses class_weights OR
  the sampler, never both at once (avoid double-correcting for
  imbalance).
- `models/baseline_cnn.pt` — trained once, but only on a dev-mode
  subset (`--max-samples 200`), not the full dataset yet.
- `src/evaluate.py` — NOT YET BUILT.
- Transfer model — NOT YET TRAINED on the full dataset.
- Gradio app — NOT YET BUILT.
- Deployment — NOT DONE.

## Known environment gotchas
- Windows + relative imports: `from . import config` fails when running
  `python src/whatever.py` directly (as opposed to `python -m src.whatever`).
  Every `src/` file needs the try/except ImportError fallback pattern
  already used in the existing files. Follow that same pattern in any
  new file.
- Imbalance ratio in the full dataset: 36.1 (max class count / min class
  count). The weighted sampler activates automatically once this ratio
  exceeds `config.IMBALANCE_RATIO_THRESHOLD`.
- CPU training time estimate on the full dataset: baseline CNN ~1–2
  min/epoch, transfer ResNet18 ~2–4 min/epoch. Use `--max-samples 200`
  for fast iteration/smoke-testing before committing to a full run.
- `data/` (raw/processed/.hf_cache) holds the actual dataset files and
  HF's download cache — large, regenerates automatically from
  `HF_DATASET_ID` in config.py. Should be gitignored. Never needs to be
  reviewed, shared, or zipped up.

## Working agreement
- Build one file at a time. Stop for review after each file before
  moving to the next.
- Never claim a file was written without proving it: right after any
  Write, run `cat <file>` and `git status` in the same turn and show
  the output as proof. Don't say "done" without evidence.
- Don't touch files marked "done" above without asking first, even to
  fix something small — flag the issue and wait for a go-ahead, UNLESS
  it's a hard blocker (e.g. the dataset is genuinely broken), in which
  case fix it but say so clearly and explain why. Never silently edit a
  file marked done.
