"""Validation OA in percent, rounded exactly as the experiment entry expects."""
import torch
from .classification_metrics import ClassificationMetrics


def validate(model, loader, criterion=None, num_classes=None, return_metrics=False):
    """Keep the default percent OA API; optionally return loss/accuracy/matrix."""
    model.eval()
    device = next(model.parameters()).device
    statistics = ClassificationMetrics(num_classes)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets) if criterion is not None else None
            statistics.update(outputs, targets, loss)
    result = statistics.result()
    return result if return_metrics else round(result['accuracy'] * 100, 2)
