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
        mask_flat = mask.flatten().bool()
        batch_size, seq_len, _ = x.size()
        norm = torch.norm(grad_output)
        scale = ctx.threshold / norm if norm > ctx.threshold else 1.
        target = (x - scale * grad_output).view(batch_size * seq_len, seq_len)
        target = torch.masked_select(target, mask_flat[:, None]).view(-1, seq_len)
        projected = project_onto_knapsack_constraint_batch(target)
        output = torch.where(mask[:, :, None].bool(), x, torch.zeros_like(x))
        output.view(-1, seq_len)[mask_flat] -= projected
        return output, None, None


def differentiable_eisner(x, mask, threshold=1.):
    return DifferentiableEisner.apply(x, mask, threshold)


def test():
    # ROOT this is a pen
    device = torch.device(0)
    scores = torch.tensor([
        [
            [0., 0., 0., 0., 0., 0.],  # ROOT
            [0., 0., 1., 0., 0., 0.],  # this
            [1., 0., 0., 0., 0., 0.],  # is
            [0., 0., 0., 0., 1., 0.],  # a
            [0., 0., 1., 0., 0., 0.],  # pen
            [0., 0., 0., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0., 0., 0.],  # ROOT
            [0., 0., 1., 0., 0., 0.],  # this
            [1., 0., 0., 0., 0., 0.],  # is
            [0., 0., 0., 0., 0., 1.],  # a
            [0., 0., 0., 0., 0., 1.],  # great
            [0., 0., 1., 0., 0., 0.],  # pen
        ],
    ], dtype=torch.float, requires_grad=True, device=device)

    mask = torch.ones((2, 6), device=device).long()
    mask[0, -1] = 0

    dep = differentiable_eisner(scores, mask)
    dep.sum().backward()
    print(dep)
    print(scores.grad)

# test()
