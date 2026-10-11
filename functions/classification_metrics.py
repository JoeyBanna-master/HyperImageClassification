"""Streaming classification diagnostics, without retaining batch predictions."""

import torch


class ClassificationMetrics:
    """Accumulate a sample-weighted mean loss and true-row/predicted-column counts.

    ``mean_loss`` must be a scalar batch mean (the project's CrossEntropyLoss).
    Accuracy is returned as a fraction; the matrix uses zero-based class indices.
    """

    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self.confusion = None
        self.loss_sum = None
        self.loss_samples = 0
        self.num_samples = 0

    def update(self, logits, targets, mean_loss=None):
        if self.num_classes is None:
            self.num_classes = logits.shape[1]
        if self.confusion is None:
            self.confusion = torch.zeros(
                (self.num_classes, self.num_classes), dtype=torch.int64, device=logits.device)
            self.loss_sum = torch.zeros((), dtype=torch.float64, device=logits.device)
        predictions = logits.detach().argmax(dim=1)
        indices = targets.to(torch.int64) * self.num_classes + predictions
        self.confusion += torch.bincount(
            indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        count = targets.numel()
        self.num_samples += count
        if mean_loss is not None:
            self.loss_sum += mean_loss.detach().to(torch.float64) * count
            self.loss_samples += count

    def result(self):
        if self.num_samples == 0:
            raise ValueError('Loader produced no samples; check batch_size/drop_last.')
        confusion = self.confusion.cpu()
        return {
            'loss': self.loss_sum.item() / self.loss_samples if self.loss_samples else None,
            'accuracy': confusion.diagonal().sum().item() / self.num_samples,
            'num_samples': self.num_samples,
            'confusion_matrix': confusion.tolist(),
        }
