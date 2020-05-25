
import torch


def gather_row(x, indices):
    assert len(x.size()) == 3 and len(indices.size()) == 2, \
            'not supported input tensor shape'
    # assert x.size()[:-1] == indices.size(), \
    #         'the first two dimensions of `x` must be same as the shape of `indices`'
    batch_size, sequence_size, hidden_size = x.size()
    indices += torch.arange(
            0, batch_size * sequence_size, sequence_size).to(x.device)[:, None]

    out = x.view((batch_size * sequence_size, hidden_size))
    out = out.index_select(0, indices.flatten())
    out = out.reshape(indices.size() + (hidden_size,))
    return out

