from dataclasses import dataclass
from typing import List
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from sklearn.preprocessing import LabelBinarizer
from transformers import CamembertTokenizer, CamembertModel

class FTVectorizer():
    def __init__(self):
        self.label_binarizer = LabelBinarizer()
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.max_length = 128  # Appropriate length for transformer models
        
    def fit(self, train_texts: List[str], train_labels: List[str]):
        # Fit label binarizer
        self.label_binarizer.fit(train_labels)

    def input_size(self) -> int:
        # CamemBERT base has 768 hidden dimensions
        return 768

    def output_size(self) -> int:
        return len(self.label_binarizer.classes_)

    def text_to_indices(self, text):
        # Use CamemBERT tokenizer directly
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0]
        }

    def vectorize_input(self, texts) -> List[dict]:
        return [self.text_to_indices(text) for text in texts]

    def vectorize_labels(self, labels) -> List[torch.Tensor]:
        vects = self.label_binarizer.transform(labels)
        return [torch.from_numpy(vect).float() for vect in vects]

    def devectorize_labels(self, prediction_vects):
        return self.label_binarizer.inverse_transform(prediction_vects)

    def batch_collate_fn(self, batch_list):
        input_vects, label_vects = tuple(zip(*batch_list))
        batch = {
            'input_ids': torch.stack([item['input_ids'] for item in input_vects]),
            'attention_mask': torch.stack([item['attention_mask'] for item in input_vects])
        }
        
        if label_vects[0] is not None:
            batch['label_vects'] = torch.stack(label_vects).float()
        return batch

@dataclass
class HyperParameters:
    batch_size: int = 16
    learning_rate: float = 5e-5
    max_epochs: int = 10
    dropout: float = 0.1
    hidden_size: int = 768  # Match CamemBERT's hidden size
    es_monitor: str = 'val_loss'
    es_mode: str = 'min'
    es_patience: int = 3
    es_min_delta: float = 0.01
    ckpt_monitor: str = 'val_loss'
    ckpt_mode: str = 'min'
    warmup_steps: int = 100
    weight_decay: float = 0.01

HP = HyperParameters()

class FTClassifier(pl.LightningModule):
    def __init__(self, vectorizer: FTVectorizer):
        super().__init__()
        self.vectorizer = vectorizer
        self.output_size = vectorizer.output_size()
        
        # Load pre-trained CamemBERT model
        self.camembert = CamembertModel.from_pretrained("camembert-base")
        
        # Freeze earlier layers
        modules = [self.camembert.embeddings, *self.camembert.encoder.layer[:8]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(HP.hidden_size, HP.hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(HP.dropout),
            torch.nn.Linear(HP.hidden_size // 2, self.output_size)
        )
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, batch):
        # Get CamemBERT embeddings and use CLS token
        outputs = self.camembert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_dict=True
        )
        
        # Get CLS token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        return logits

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.loss_fn(y_hat, batch['label_vects'])
        
        # Calculate training accuracy
        predictions = torch.argmax(y_hat, dim=1)
        labels = torch.argmax(batch['label_vects'], dim=1)
        acc = (predictions == labels).float().mean()
        
        self.log_dict({
            'train_loss': loss,
            'train_acc': acc
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=HP.learning_rate,
            weight_decay=HP.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=HP.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def validation_step(self, batch, batch_ix):
        y_hat = self.forward(batch)
        loss = self.loss_fn(y_hat, batch['label_vects'])
        
        # Calculate validation metrics
        predictions = torch.argmax(y_hat, dim=1)
        labels = torch.argmax(batch['label_vects'], dim=1)
        acc = (predictions == labels).float().mean()
        
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        y_hat = self.forward(batch)
        y_hat = F.softmax(y_hat, dim=1)
        return y_hat