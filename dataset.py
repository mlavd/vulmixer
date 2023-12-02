"""FIRE JSONL Dataset.

This module provides a JSONLDataModule for loading the MLAVD datasets. It is
used through the curriculum learning datasets, but may be used independently
with another Pytorch Lightning training module.
"""
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch


class JSONLDataModule(pl.LightningDataModule):
    def __init__(self, config, batch_size, num_workers, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.train_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _build_dataset(self, split):
        return JSONLDataset(path=self.config[split])

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            self.train_dataset = self._build_dataset('train')
            self.valid_dataset = self._build_dataset('valid')
        if stage in (None, 'test'):
            self.test_dataset = self._build_dataset('test')
    
    def train_labels(self) -> np.array:
        if not self.train_dataset: self.setup('fit')
        return self.train_dataset.labels()

    def _build_dataloader(self, dataset, batch_size, shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def get_class_weights(self):
        if not self.train_dataset: self.setup('fit')
        w = self.train_dataset.data.shape[0] / np.bincount(self.train_dataset.data.target)
        return torch.tensor(w).to(torch.float32)

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.valid_dataset, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.test_dataset, self.batch_size)


class JSONLDataset(Dataset):
    def __init__(self, path, **kwargs) -> None:
        super().__init__()
        self.data = pd.read_json(path, lines=True, dtype_backend='pyarrow')
    
    def labels(self) -> np.array:
        return self.data['target'].values

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data.iloc[index]
        return {
            'index': index,
            'input': fields.func,
            'target': fields.target,
        }
