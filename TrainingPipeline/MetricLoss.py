import torch
import torch.nn as nn


class MetricLoss(nn.Module):
    def __init__(self, metric_fn, base_loss=nn.CrossEntropyLoss(), alpha=0.5):
        """
        Combines CrossEntropy Loss with the value of a passed Metric
        Args:
            base_loss: Base loss function (e.g., CrossEntropyLoss)
            metric_fn: sklearn metric function
            alpha: Weight for metric component (1-alpha for base loss)
        """
        super().__init__()
        self.base_loss = base_loss
        self.metric_fn = metric_fn
        self.alpha = alpha

    def forward(self, outputs, targets):
        base_loss_val = self.base_loss(outputs, targets)

        # Calculate metric on current batch (may be unstable)
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()
            try:
                metric_val = self.metric_fn(targets_np, preds)
            except:
                metric_val = 0.0

        return (1 - self.alpha) * base_loss_val + self.alpha * (1 - metric_val)
