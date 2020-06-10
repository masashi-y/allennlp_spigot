import torch
from spigot.algorithms.eisner import eisner
from spigot.algorithms.krucker import project_onto_knapsack_constraint_batch


class DifferentiableEisner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, threshold):
        results = eisner(x, mask)
        ctx.save_for_backward(x, mask)
        ctx.threshold = threshold
        return results.float()

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        batch_size, seq_len, _ = x.size()

        norm = torch.norm(grad_output)
        scale = ctx.threshold / norm if norm > ctx.threshold else 1.
        # single headed constraints (all heads scores for a single token must sum to one)
        target = (x - scale * grad_output).view(batch_size * seq_len, seq_len)
        # do not perform projection on masked items
        projected = project_onto_knapsack_constraint_batch(
                target,
                mask=mask.flatten().bool(),
                padding=0.)
        output = torch.where(mask[:, :, None].bool(), x, torch.zeros_like(x))
        output -= projected.view_as(x)
        return output, None, None

def differentiable_eisner(x, mask, threshold=1.):
    return DifferentiableEisner.apply(x, mask, threshold)


