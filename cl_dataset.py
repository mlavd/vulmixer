"""FIRE Curriculum Learning Dataset Loading.

This module supports multiple forms of curriculum learning for training the
MLAVD models. Configuration files are not provided, but are easy to produce.
Only the `NoCLDataset` has been thoroughly tested.
"""
from dataset import JSONLDataModule, JSONLDataset
import lizard
import numpy as np
import torch

class CurriculumDataset(JSONLDataset):
    def __init__(self, path, cl_config) -> None:
        super().__init__(path=path)
        self.cl_config = cl_config
        self.fit()
        self.epoch = 0
        self.data_full = None


    def update_epoch(self, epoch):
        self.epoch = epoch
        if 'schedule' not in self.cl_config: return

        if self.data_full is None:
            # Save the items once.
            self.data_full = self.data.copy()
            self.weights_full = self.weights.clone()

        if self.cl_config.schedule == 'linear':
            split = (epoch // self.cl_config.linear_steps + 1) * self.cl_config.linear_split
            k = int(self.data_full.shape[0] * split)
            print(f'Epoch {epoch}, using {split:.0%} of data ({k:,d} samples).')
            _, indices = torch.topk(self.weights_full, k, largest=False, sorted=False)
            self.data = self.data_full.iloc[indices.numpy()]
            self.weights = self.weights_full[indices]
        elif self.cl_config.schedule == 'none':
            pass
        else:
            raise NotImplementedError()

    def get_weight(self, index, fields):
        raise NotImplementedError()

    def __getitem__(self, index):
        fields = self.data.iloc[index]
        weight = self.get_weight(index, fields)
        m = max(1, self.cl_config.weight)
        weighted = (m - (1 - weight) * self.cl_config.weight)
        # print(weight, weighted)
        # assert weighted > 0
        return {
            'index': index,
            'input': fields.func,
            'target': fields.target,
            'weight': (1 - (1 - weight) * self.cl_config.weight), # FIXME doesn't work for values over 1
        }

class FittedDataset(CurriculumDataset):
    def get_weight(self, index, fields):
        return self.weights[index]

class CyclomaticComplexityCLDataset(FittedDataset):
    def fit(self):
        w = torch.zeros((self.data.shape[0],), dtype=float)
        for i, row in self.data.iterrows():
            result = lizard.analyze_file.analyze_source_code('arbitrary.cpp', row.func)
            complexity = np.mean([r.cyclomatic_complexity for r in result.function_list ])
            w[i] = complexity


        if w.isnan().any():
            print(f'Cyclomatic complexity could not be calculated for {w.isnan().sum():,d} samples.')
            print(f"Random values in the dataset's range will be selected for these.")
            mask = w.isnan()
            w[mask] = torch.randint(
                low=w[~mask].min().to(int),
                high=w[~mask].max().to(int),
                size=(mask.sum().item(),)
            ).to(float)

        self.weights = 1 - w / (w.max() + 1)

class TokenCountCLDataset(FittedDataset):
    def fit(self):
        w = torch.zeros((self.data.shape[0],), dtype=float)
        for i, row in self.data.iterrows():
            result = lizard.analyze_file.analyze_source_code('arbitrary.cpp', row.func)
            complexity = np.mean([r.token_count for r in result.function_list ])
            w[i] = complexity


        if w.isnan().any():
            print(f'Cyclomatic complexity could not be calculated for {w.isnan().sum():,d} samples.')
            print(f"Random values in the dataset's range will be selected for these.")
            mask = w.isnan()
            w[mask] = torch.randint(
                low=w[~mask].min().to(int),
                high=w[~mask].max().to(int),
                size=(mask.sum().item(),)
            ).to(float)

        self.weights = 1 - w / (w.max() + 1)

class NoCLDataset(FittedDataset):
    def fit(self):
        self.weights = torch.ones((self.data.shape[0],))

class LengthCLDataset(FittedDataset):
    def fit(self):
        lengths = self.data.func.map(len)
        self.weights = torch.as_tensor(1 - lengths / (lengths.max() + 1))
        # print(lengths.values)
        # print(self.weights.numpy())
        # exit(-1)

class RandomCLDataset(FittedDataset):
    def fit(self):
        self.weights = torch.rand((self.data.shape[0],))


class CurriculumDataModule(JSONLDataModule):
    DATASET_CLASSES = {
        'length': LengthCLDataset,
        'tokens': TokenCountCLDataset,
        # 'ifs': None,
        'cyclomatic': CyclomaticComplexityCLDataset,
        'random': RandomCLDataset,
        'none': NoCLDataset,
    }

    def __init__(self, cl_config, data_config, batch_size, num_workers, **kwargs):
        super().__init__(
            config=data_config,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs)
        self.cl_config = cl_config

    def train_dataloader(self):
        # print(self.__dict__.keys())
        # print(self.trainer)
        # print('-' * 50, 'New Loader!' * 20)
        if hasattr(self.trainer, 'current_epoch'):
            self.train_dataset.update_epoch(self.trainer.current_epoch)
        return self._build_dataloader(self.train_dataset, self.batch_size, shuffle=True)

    def _build_dataset(self, split):
        if split == 'train':
            if self.cl_config.type in self.DATASET_CLASSES:
                return self.DATASET_CLASSES[self.cl_config.type](
                    path=self.config[split],
                    cl_config=self.cl_config,
                )
            else:
                raise NotImplementedError(f'Invalid CL type: {self.cl_config.type}')
        else:
            return NoCLDataset(
                path=self.config[split],
                cl_config=self.cl_config,
            )