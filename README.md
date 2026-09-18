# Plant Disease Detection: End-to-End Image Classifier

A robust deep learning pipeline for plant disease detection using leaf images from the PlantVillage dataset via Hugging Face Datasets.

## Project Structure

```
plant-disease-classifier/
  data/            - Dataset download/prep scripts, not raw data committed
  notebooks/       - EDA only, not the main deliverable
  src/
    data.py        - Dataset/DataLoader, transforms, train/val/test split
    models.py      - CNN-from-scratch & transfer learning (ResNet18/EfficientNet)
    train.py       - Training loop, checkpointing, early stopping, logging
    evaluate.py    - Accuracy, precision/recall/F1, confusion matrix, plots
  api/
    main.py        - FastAPI app: /predict (image upload) -> class + confidence, /health
    schemas.py     - Pydantic request/response models
  tests/           - Unit tests for data pipeline, model forward pass, API endpoint
  Dockerfile
  requirements.txt
  README.md
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download and prepare the dataset:
   ```bash
   python src/data.py
   ```

3. Train the models:
   ```bash
   python src/train.py --model baseline
   python src/train.py --model transfer
   ```

4. Evaluate the models:
   ```bash
   python src/evaluate.py
   ```

5. Run the API locally:
   ```bash
   uvicorn api.main:app --reload
   ```

## Model Comparison & Recommendation
*To be filled out after training and evaluation.*
