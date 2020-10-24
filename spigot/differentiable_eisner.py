import torch

from spigot.algorithms.eisner import eisner
from spigot.algorithms.krucker import project_onto_knapsack_constraint_batch


class DifferentiableEisner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, threshold):
        results = eisner(x, mask)
        ctx.save_for_backward(results, mask)
        ctx.threshold = threshold
        return results.float()

    @staticmethod
    def backward(ctx, grad_output):
        predicted, mask = ctx.saved_tensors
        batch_size, seq_len, _ = predicted.size()

        norm = torch.norm(grad_output, dim=-1)
        scale = torch.ones_like(norm)
        cond = norm > 1.0
        scale[cond] = 1.0 / norm[cond]

        # single headed constraints (all heads scores for a single token must sum to one)
        target = (predicted - scale * grad_output).view(batch_size * seq_len, seq_len)
        # do not perform projection on masked items
        projected = project_onto_knapsack_constraint_batch(
            target, mask=mask.flatten().bool(), padding=0.0
        )
        output = torch.where(
            mask[:, :, None].bool(), predicted, torch.zeros_like(predicted)
        )
        output -= projected.view_as(predicted)
        return output, None, None


def differentiable_eisner(x, mask, threshold=1.0):
    return DifferentiableEisner.apply(x, mask, threshold)
