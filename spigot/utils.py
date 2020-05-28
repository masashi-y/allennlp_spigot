
import torch


def gather_row(x, indices):
    """Kind of a broadcast version of `torch.gather` function
    Currently this support for inputs `x` with 3 dimensions and
    `indices` with 2 dimensions.

    Example:
    >>> x = torch.tensor([
    ...     [[1, 2],
    ...      [3, 4]],
    ...     [[5, 6]
    ...      [7, 8]]
    ... ])
    >>> indices = torch.tensor(
    ...     [[0, 0],
    ...      [1, 0]]
    ... )
    >>> gather_row(x, indices)
    torch.tensor([
        [[1, 2],
         [1, 2]]
        [[7, 8],
         [5, 6]]
    ])
    """
    assert len(x.size()) == 3 and len(indices.size()) == 2, \
            'not supported input tensor shape'
    batch_size, sequence_size, hidden_size = x.size()
    indices += torch.arange(
            0, batch_size * sequence_size, sequence_size).to(x.device)[:, None]

    out = x.view((batch_size * sequence_size, hidden_size))
    out = out.index_select(0, indices.flatten())
    out = out.reshape(indices.size() + (hidden_size,))
    return out

