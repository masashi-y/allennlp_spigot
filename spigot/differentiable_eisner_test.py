import torch
from spigot import differentiable_eisner

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


if __name__ == "__main__":
    test()
