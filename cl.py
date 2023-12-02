"""FIRE Training Module.

This module provides the Pytorch Lighning module used to train all of the
models.
"""
from torchmetrics.classification.stat_scores import BinaryStatScores
import lightning.pytorch as pl
import psutil
import torch
import torch.nn.functional as F
import torchmetrics

class CurriculumLearning(pl.LightningModule):
    def __init__(self, model, transform, config, device,
        weights, model_config, calculate_metrics=True, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.transform = transform
        self.config = config
        self.num_classes = len(weights)
        self.on_device = device # FIXME Use self.device instead of this
        self.weights = weights.to(device)
        self.model_config = model_config
        self.calculate_metrics = calculate_metrics

        self.test_preds = {}
        self.train_acc, self.train_ap, self.train_f1 = self._metrics(self.num_classes)
        self.valid_acc, self.valid_ap, self.valid_f1 = self._metrics(self.num_classes)
        self.test_acc, self.test_ap, self.test_f1 = self._metrics(self.num_classes)

        self.max = {
            'max_train_acc': torch.as_tensor([0]).to(device),
            'max_train_ap': torch.as_tensor([0]).to(device),
            'max_train_f1': torch.as_tensor([0]).to(device),
            'max_val_acc': torch.as_tensor([0]).to(device),
            'max_val_ap': torch.as_tensor([0]).to(device),
            'max_val_f1': torch.as_tensor([0]).to(device),
        }
    
    def _log_max(self, metric, name):
        val = metric.compute().to(self.on_device)
        self.max[name] = torch.maximum(val, self.max[name])
        self.log(name, self.max[name], on_epoch=True, logger=True)
        return val
    
    def on_train_epoch_end(self):
        # FIXME Clean this up
        if self.calculate_metrics and not self.trainer.sanity_checking:
            self._log_max(self.train_acc, 'max_train_acc')
            self._log_max(self.train_ap, 'max_train_ap')
            self._log_max(self.train_f1, 'max_train_f1')

    def on_validation_epoch_end(self):
        # FIXME Clean this up
        if self.calculate_metrics and not self.trainer.sanity_checking:
            self._log_max(self.valid_acc, 'max_val_acc')
            self._log_max(self.valid_ap, 'max_val_ap')
            self._log_max(self.valid_f1, 'max_val_f1')

    def _metrics(self, num_classes):
        return (
            torchmetrics.Accuracy(task='multiclass', num_classes=num_classes),
            torchmetrics.AveragePrecision(task='binary', num_classes=num_classes),
            torchmetrics.F1Score(task='binary', num_classes=2)
        )
    
    def update_metrics(self, metrics, logits, targets):
        for metric in metrics:
            if isinstance(metric, BinaryStatScores):
                metric(logits.argmax(dim=-1).detach(), targets)
            elif isinstance(metric, torchmetrics.classification.average_precision.BinaryAveragePrecision):
                metric(logits[:, 1].detach(), targets)
            else:
                metric(logits.detach(), targets)
    
    def shared_step(self, batch, metrics):
        # Get vars
        inputs = batch['input']
        targets = batch['target'].to(self.on_device)
        weights = batch['weight'].to(self.on_device)

        # Apply transform
        inputs = self.transform(inputs)
        inputs = torch.tensor(inputs).to(self.on_device)

        # self.model.to(self.on_device)

        # Predict and update the loss.
        logits = self.model(inputs)
        if len(logits.shape) == 1: logits = logits[None, :]
        loss = F.cross_entropy(logits, targets, weight=self.weights, reduction='none')

        if loss.isnan().any():
            print('NaN loss!')
            exit(-1)

        # Add the curriculum weighting
        loss = loss * weights
        loss = loss.mean()

        # Update the metrics
        if self.calculate_metrics:
            self.update_metrics(metrics, logits, targets)

        return { 'loss': loss,  'all': logits, }

    def log_metrics(self, split, loss, acc, ap, f1, maxap=None, on_step=False):
        self.log(f'{split}_loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)

        if self.calculate_metrics:
            self.log(f'{split}_acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'{split}_ap', ap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{split}_f1', f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, [self.train_acc, self.train_ap, self.train_f1])
        self.log_metrics('train', loss=results['loss'], acc=self.train_acc, ap=self.train_ap, f1=self.train_f1, on_step=True)

        # Periodically monitor the RAM
        if not batch_idx % 100:
            self.log('ram', psutil.virtual_memory()[2], on_step=True, prog_bar=True, logger=True)

        return results
        
    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch, [self.valid_acc, self.valid_ap, self.valid_f1])
        self.log_metrics('val', loss=results['loss'], acc=self.valid_acc, ap=self.valid_ap, f1=self.valid_f1)
        return results
        
    def test_step(self, batch, batch_idx):
        results = self.shared_step(batch, [self.test_acc, self.test_ap, self.test_f1])
        self.log_metrics('test', loss=results['loss'], acc=self.test_acc, ap=self.test_ap, f1=self.test_f1)

        results['all'] = torch.softmax(results['all'], dim=-1)
        for i, logit in zip(batch['index'], results['all']):
            self.test_preds[i.cpu().item()] = logit.cpu().numpy()[1]

        return results

    def configure_optimizers(self):
        # FIXME Move optimizers to individual models?
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.model_config.optimizer.lr),
            betas=self.model_config.optimizer.betas,
            eps=float(self.model_config.optimizer.eps),
        )
