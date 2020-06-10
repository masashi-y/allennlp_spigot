from typing import Optional, List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("semantic_dependencies")
class SemanticDependenciesF1(Metric):
    def __init__(self) -> None:
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._pred_sum = 0.0
        self._true_sum = 0.0

    def __call__(  # type: ignore
        self,
        predicted_arc_tags: torch.Tensor,
        gold_arc_tags: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        predicted_arc_tags, gold_arc_tags, mask= self.detach_tensors(
            predicted_arc_tags, gold_arc_tags, mask)

        if mask is None:
            mask = torch.ones_like(predicted_arc_tags).bool()

        predicted_arc_tags = predicted_arc_tags.long()
        gold_arc_tags = gold_arc_tags.long()
        predicted_arcs = (predicted_arc_tags >= 0).long()
        gold_arcs = (gold_arc_tags >= 0).long()
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        labeled_correct = predicted_arc_tags.eq(gold_arc_tags) \
                .logical_and(predicted_arcs) \
                .logical_and(gold_arcs) \
                .long() * mask
        unlabeled_correct = predicted_arcs * gold_arcs * mask

        self._labeled_correct += labeled_correct.sum()
        self._unlabeled_correct += unlabeled_correct.sum()
        self._pred_sum += (predicted_arcs * mask).sum()
        self._true_sum += (gold_arcs * mask).sum()

    def get_metric(self, reset: bool = False):
        labeled_precision = _prf_divide(self._labeled_correct, self._pred_sum)
        labeled_recall = _prf_divide(self._labeled_correct, self._true_sum)
        labeled_fscore = (2 * labeled_precision * labeled_recall) / (labeled_precision + labeled_recall)
        labeled_fscore[self._labeled_correct == 0] = 0.0
        unlabeled_precision = _prf_divide(self._unlabeled_correct, self._pred_sum)
        unlabeled_recall = _prf_divide(self._unlabeled_correct, self._true_sum)
        unlabeled_fscore = (2 * unlabeled_precision * unlabeled_recall) / (unlabeled_precision + unlabeled_recall)
        unlabeled_fscore[self._unlabeled_correct == 0] = 0.0

        if reset:
            self.reset()
        return {
            "labeled_precision": labeled_precision.item(),
            "labeled_recall": labeled_recall.item(),
            "labeled_fscore": labeled_fscore.item(),
            "unlabeled_precision": unlabeled_precision.item(),
            "unlabeled_recall": unlabeled_recall.item(),
            "unlabeled_fscore": unlabeled_fscore.item()
        }

    @overrides
    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._pred_sum = 0.0
        self._true_sum = 0.0


def _prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result
