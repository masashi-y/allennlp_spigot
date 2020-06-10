
import torch
from allennlp.nn.util import tiny_value_of_dtype


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


def masked_gumbel_softmax(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1, tau: float = 1,
) -> torch.Tensor:
    """
    `torch.nn.functional.gumbel_softmax(vector)` does not work if some elements of `vector`
    should be masked.  This performs a gumbel_softmax on just the non-masked portions of `vector`.
    Passing `None` in for the mask is also acceptable; you'll just get a regular gumbel softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    """
    if mask is None:
        result = torch.nn.functional.gumbel_softmax(vector, dim=dim, tau=tau)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        result = torch.nn.functional.gumbel_softmax(vector * mask, dim=dim, tau=tau)
        result = result * mask
        result = result / (
            result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
        )
    return result


def kl_divergence(qs, ps, mask):
    eps = tiny_value_of_dtype(ps.dtype)
    outputs = torch.sum(ps * (torch.log(ps + eps) - torch.log(qs + eps)), dim=-1)
    return (outputs * mask).mean()
