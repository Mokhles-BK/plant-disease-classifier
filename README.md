# Plant Disease Classifier

An end-to-end deep learning app that identifies plant diseases from leaf photos. Upload a photo, get the top-3 predicted diagnoses with confidence scores.

Built with PyTorch (CNN from scratch + transfer learning on ResNet18) and served through a Gradio web app.

## Results

Trained on the [PlantVillage dataset](https://huggingface.co/datasets/GVJahnavi/PlantVillage_dataset) (38 classes, 43,503 training images, 10,878 test images), with a severe class imbalance (36:1 largest-to-smallest class ratio) handled via a weighted sampler. Both models below were trained on the full dataset and evaluated on the same held-out test split.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **CNN from scratch** | **97.70%** | **96.85%** | **97.70%** |
| Transfer learning (ResNet18) | 95.42% | 93.95% | 95.44% |

**The from-scratch CNN outperformed ResNet18 transfer learning on this task.** A plausible reason: the images are small (256x256, resized to 224) and domain-specific (leaf textures and color patterns), which differs quite a bit from the natural photos ResNet18's ImageNet pretraining was built on. A compact model trained directly on the target domain, with a large enough dataset (43k+ images) and enough epochs, was able to learn more task-specific features than a frozen pretrained backbone could offer here.

Full per-class precision/recall/F1 and confusion matrices for both models are in [`reports/`](reports/).

## Project structure

```
plant-disease-classifier/
  src/
    config.py       - central config: paths, hyperparameters, dataset ID
    data.py         - dataset loading, stratified split, transforms, DataLoaders
    models.py       - CNN-from-scratch and ResNet18 transfer learning
    train.py        - training loop, checkpointing, early stopping
    evaluate.py      - accuracy, precision/recall/F1, confusion matrix
    utils.py        - shared helpers (seeding, device selection, class weights)
  app.py             - Gradio web app (upload image -> top-3 predictions)
  reports/           - saved metrics (JSON) and confusion matrices (PNG)
  models/            - trained checkpoints (not committed, see below)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train a model:
```bash
python src/train.py --model baseline    # CNN from scratch
python src/train.py --model transfer    # ResNet18 transfer learning
```

Evaluate a trained checkpoint:
```bash
python src/evaluate.py --ckpt models/baseline_cnn.pt
python src/evaluate.py --ckpt models/transfer_resnet18.pt
```

Run the web app locally:
```bash
python app.py
```
Then open `http://127.0.0.1:7860` in a browser.

## Notes

- Model checkpoints (`models/*.pt`) aren't committed to this repo (too large for plain git). Train from scratch with the commands above, or the checkpoints can be provided separately.
- Training used a GPU (Google Colab, T4) for both full-dataset runs; the codebase also runs on CPU (slower).
- The deployed app uses the CNN-from-scratch checkpoint (better test-set performance).
- Live demo: *deployment pending.*
