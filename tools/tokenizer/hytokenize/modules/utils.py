import torch
from collections import OrderedDict


def exists(val):
    return val is not None


def random_masking(mask, mask_prob):
    assert mask.ndim == 2
    lens = mask.shape[-1]
    rand = torch.randn(mask.shape, device=mask.device)
    rand[:, 0] = -torch.finfo(rand.dtype).max # Ignore the first item
    num_mask = min(int(lens * mask_prob), lens - 1)
    indices = rand.topk(num_mask, dim=-1).indices
    new_mask = ~torch.zeros(mask.shape, device=mask.device).scatter(1, indices, 1.).bool()
    return new_mask


def get_mask_from_lengths(lengths, max_len=None, r=1, random_mask=0.):
    if max_len is None:
        max_len = torch.max(lengths).item()
    if max_len % r != 0:
        max_len = max_len + r - max_len % r
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(lengths.device))
    mask = (ids < lengths.unsqueeze(1)).bool()
    if random_mask > 0.:
        mask = mask.logical_and(random_masking(mask, random_mask))
    return mask


def get_pad_mask_from_lengths(lengths, max_len=None, r=1):
    """Get mask, where padding part has value true."""
    return ~get_mask_from_lengths(lengths, max_len, r)


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner
