# Offline Text Summarizer

A complete end-to-end NLP project for text summarization that runs fully offline.

## Features
- **Extractive Summarization**: TextRank algorithm implemented from scratch (TF-IDF + PageRank).
- **Abstractive Summarization**: Transformer-based generation using local T5 weights.
- **Offline ROUGE Evaluation**: Calculate precision, recall, and F-measure without API calls.
- **Dataset Support**: Load and preprocess CNN/DailyMail dataset from local disk.
- **Fine-tuning**: Scripts provided to fine-tune transformers on custom local data.

## Project Structure
```text
text_summarizer/
├── data/              # Place your local datasets here (e.g., cnn_dailymail/)
├── models/            # Downloaded model weights (t5, bart, etc.)
├── extractive/        # TextRank implementation
├── abstractive/       # Transformer wrapper and trainer
├── evaluation/        # ROUGE metrics
├── main.py            # CLI entry point
├── setup_offline.py   # One-time script to download models
└── requirements.txt   # Dependencies
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models (While Online)
Run the script to download weights for local usage:
```bash
python setup_offline.py
```

### 3. Dataset Download (Optional)
Download the CNN/DailyMail dataset (CSV format) from [Kaggle](https://www.kaggle.com/datasets/gowrishankarp/anonymous-github-users-dataset) or Hugging Face. Place the `train.csv` inside `data/cnn_dailymail/`.

## Usage

### Run the Summarizer CLI
```bash
python main.py
```
You can then paste any text and choose the summarization type.

### Options
- `--type [extractive|abstractive|both]`: Choose summarization method.
- `--sentences 5`: Set summary length for extractive mode.
- `--max_len 100`: Set max length for abstractive mode.

### Running fine-tuning
To fine-tune on your local dataset:
```python
from data.preprocessing import LocalDatasetLoader
from abstractive.trainer import SummarizationTrainer

# Load your local data
loader = LocalDatasetLoader()
df = loader.load_from_csv("train.csv", limit=1000)
df = loader.preprocess(df)

# Initialize trainer and train
trainer = SummarizationTrainer(model_checkpoint="t5-small")
# dataset conversion logic...
trainer.train(tokenized_datasets)
```

## Evaluation
The system compares execution times and provides ROUGE scores when evaluated against a reference summary.
