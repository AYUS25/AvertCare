"""
FT-Transformer (Feature Tokenizer + Transformer Encoder for Tabular Data)
PyTorch Implementation for Readmission Risk Prediction
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class TabularDataset(Dataset):
    """
    PyTorch Dataset for Tabular Data (Numerical + Categorical).
    """
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        return self.X_num[idx], self.X_cat[idx]


class FeatureTokenizer(nn.Module):
    """
    Feature Tokenizer converts numerical and categorical features into d_token dimensional vectors.
    Includes trainable [CLS] token.
    """
    def __init__(self, n_num_features, cat_cardinalities, d_token):
        super().__init__()
        self.n_num = n_num_features
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token

        # Numerical feature weight & bias per feature
        self.num_weights = nn.Parameter(torch.randn(n_num_features, d_token) * 0.01)
        self.num_biases = nn.Parameter(torch.zeros(n_num_features, d_token))

        # Categorical feature embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card + 1, embedding_dim=d_token)
            for card in cat_cardinalities
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.01)

    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0]

        # Numerical tokenization: x_num shape (B, N_num) -> (B, N_num, d_token)
        num_tokens = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0)

        # Categorical tokenization: x_cat shape (B, N_cat) -> (B, N_cat, d_token)
        cat_tokens_list = []
        for j in range(self.n_cat):
            cat_tokens_list.append(self.cat_embeddings[j](x_cat[:, j]).unsqueeze(1))
        cat_tokens = torch.cat(cat_tokens_list, dim=1) if cat_tokens_list else torch.empty(batch_size, 0, self.d_token, device=x_num.device)

        # Expand CLS token to batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate: [CLS, Num_1...Num_12, Cat_1...Cat_22]
        x_tokens = torch.cat([cls_tokens, num_tokens, cat_tokens], dim=1)
        return x_tokens


class FTTransformerModel(nn.Module):
    """
    FT-Transformer Model Architecture.
    Feature Tokenizer -> Transformer Encoder -> CLS Head -> Sigmoid probability.
    """
    def __init__(
        self,
        n_num_features=12,
        cat_cardinalities=[4, 8, 6, 2, 2, 9, 9, 9, 11, 2, 4, 4, 4, 4, 4, 11, 4, 17, 4, 6, 4, 4],
        d_token=64,
        n_layers=3,
        n_heads=4,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num_features, cat_cardinalities, d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_num, x_cat):
        tokens = self.tokenizer(x_num, x_cat)  # (B, 1 + N_num + N_cat, d_token)
        trans_out = self.transformer(tokens)   # (B, 1 + N_num + N_cat, d_token)
        cls_out = trans_out[:, 0, :]           # (B, d_token)
        logits = self.head(cls_out)            # (B, 1)
        return logits


class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible Wrapper for FTTransformerModel.
    """
    def __init__(
        self,
        n_num_features=12,
        cat_cardinalities=[4, 8, 6, 2, 2, 9, 9, 9, 11, 2, 4, 4, 4, 4, 4, 11, 4, 17, 4, 6, 4, 4],
        d_token=64,
        n_layers=3,
        n_heads=4,
        d_ff=128,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        epochs=15,
        patience=4,
        device=None,
    ):
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None

    def _split_input(self, X):
        X_num = X[:, : self.n_num_features]
        # Shift categorical indices by +1 so that -1 (unknown category from OrdinalEncoder) maps to 0
        X_cat = np.maximum(0, X[:, self.n_num_features :].astype(int) + 1)
        return X_num, X_cat

    def fit(self, X, y, eval_set=None):
        X_num, X_cat = self._split_input(X)
        train_dataset = TabularDataset(X_num, X_cat, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_num, X_val_cat = self._split_input(X_val)
            val_dataset = TabularDataset(X_val_num, X_val_cat, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        self.model = FTTransformerModel(
            n_num_features=self.n_num_features,
            cat_cardinalities=self.cat_cardinalities,
            d_token=self.d_token,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state_dict = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0

            for bx_num, bx_cat, by in train_loader:
                bx_num, bx_cat, by = bx_num.to(self.device), bx_cat.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx_num, bx_cat)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(by)

            train_loss /= len(train_dataset)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx_num, bx_cat, by in val_loader:
                        bx_num, bx_cat, by = bx_num.to(self.device), bx_cat.to(self.device), by.to(self.device)
                        logits = self.model(bx_num, bx_cat)
                        loss = criterion(logits, by)
                        val_loss += loss.item() * len(by)

                val_loss /= len(val_dataset)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Epoch {epoch}: Early stopping triggered (best val loss: {best_val_loss:.4f})")
                        break
            else:
                best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.model.to(self.device)

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_num, X_cat = self._split_input(X)
        dataset = TabularDataset(X_num, X_cat)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs = []
        with torch.no_grad():
            for bx_num, bx_cat in loader:
                bx_num, bx_cat = bx_num.to(self.device), bx_cat.to(self.device)
                logits = self.model(bx_num, bx_cat)
                prob = torch.sigmoid(logits).cpu().numpy()
                probs.append(prob)

        p1 = np.vstack(probs).flatten()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)
