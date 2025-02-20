# 📌 Sentiment Analysis with CamemBERT

This repository contains a sentiment analysis model built using **CamemBERT** and **PyTorch Lightning**. The project includes dataset processing, a classifier, and training/evaluation scripts.

## 📂 Project Structure

```
├── classifier.py          # Defines the vectorizer and classifier using CamemBERT
├── review_dataset.py      # Handles dataset loading and preprocessing
├── run_project.py         # Main script to train and evaluate the model
├── data/                  # Folder to store dataset files (not included)
└── README.md              # Project documentation
```

## 🚀 Features

- Uses **CamemBERT** (French BERT model) for text representation.
- Implements a **PyTorch Lightning**-based classifier.
- Supports training with **early stopping** and **model checkpointing**.
- Handles **multi-class classification** using label binarization.
- Provides easy evaluation of predictions.

## 🛠 Installation

### Prerequisites

- Python 3.8+
- PyTorch & PyTorch Lightning
- Transformers (Hugging Face)
- NumPy, Pandas, Scikit-learn

## 📊 Dataset

The dataset should be in a **comma-separated (`.csv`) format** with two columns:

| polarity | text |
|----------|------|
| positive | "Nous reviendrons!" |
| negative | "On s'attendait à mieux (attente, qualité moyenne)." |
| neutral  | "Pas mal, mais sans plus." |

Place your dataset files inside the `data/` folder:

```
data/
  ├── frdataset1_train.csv  # Training dataset
  ├── frdataset1_dev.csv    # Validation dataset
  ├── frdataset1_test.csv   # (Optional) Test dataset
```

## 🔥 Training the Model

Run the training script:

```bash
python run_project.py --nruns 2 --gpu_id 0
```

Arguments:
- `--nruns`: Number of training runs (default: 1).
- `--gpu_id`: GPU device ID (default: CPU if not specified).

## 🎯 Model Architecture

- **Vectorization:** Uses CamemBERT tokenizer to convert text into token embeddings.
- **Classification Head:** A fully connected feedforward network for sentiment classification.
- **Loss Function:** Uses **CrossEntropyLoss** for multi-class classification.


## 🤖 Making Predictions

To classify new text samples:

```python
from classifier import FTVectorizer, FTClassifier
from run_project import ft_predict

vectorizer = FTVectorizer()
model = FTClassifier(vectorizer)
texts = ["Ce film est incroyable!", "Je suis déçu du produit."]
predictions = ft_predict(model, texts, accelerator="cpu", devices=None)
print(predictions)
```
