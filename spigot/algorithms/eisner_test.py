import torch
from spigot.algorithms.eisner import eisner

def test():
    # ROOT this is a pen
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
    ], dtype=torch.float)

    mask = torch.ones((2, 6)).long()
    mask[0, -1] = 0

    res = eisner(scores, mask)

    hidden_vectors = torch.arange(2 * 6 * 8).view(2, 6, 8).float()
    print(res)
    print(hidden_vectors)
    print(torch.bmm(res[:, 1:], hidden_vectors))


if __name__ == "__main__":
    test()
