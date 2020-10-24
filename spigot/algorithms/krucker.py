import torch

INF = float("inf")


def project_onto_knapsack_constraint(x, weights=None):
    """
    solves the following problem:
     \min_v  || v ||^2 s.t.
      sum_i v_i = 1 - sum_i x_i,
      - x_i <= v_i <= 1 - x_i  (i = 1, ..., n)
    and returns x + v, which is projection of x onto simplex,
    satisfying sum_i x_i + v_i = 1 and 0 <= x_i + v_i <= 1 (i = 1, ..., n).
    """
    (d,) = x.size()
    lower_bounds = -x
    upper_bounds = 1 - x
    if weights is None:
        weights = x.new_ones((d,), dtype=torch.float)
    total_weight = 1 - x.sum()
    lower_sorted, lower_sorted_indices = torch.sort(lower_bounds)
    upper_sorted, upper_sorted_indices = torch.sort(upper_bounds)
    tight_sum = torch.sum(lower_bounds * weights)

    slack_weight = 0.0
    level = k = l = 0
    left = right = -INF
    found = False

    while k < d or l < d:
        if level != 0:
            tau = (total_weight - tight_sum) / slack_weight

        if k < d:
            value_lower = lower_sorted[k]
            index_lower = lower_sorted_indices[k]
        else:
            value_lower = INF

        if l < d:
            value_upper = upper_sorted[l]
            index_upper = upper_sorted_indices[l]
        else:
            value_upper = INF

        left, right = right, min(value_lower, value_upper)

        if (
            level == 0
            and total_weight == tight_sum
            or level != 0
            and left <= tau <= right
        ):
            found = True
            break

        if value_lower < value_upper:
            tight_sum -= lower_bounds[index_lower] * weights[index_lower]
            slack_weight += weights[index_lower]
            level += 1
            k += 1
        else:
            tight_sum += upper_bounds[index_upper] * weights[index_upper]
            slack_weight -= weights[index_upper]
            level -= 1
            l += 1

    solution = x.new_full((d,), tau, dtype=torch.float)
    if not found:
        left, right = right, INF
    solution[lower_bounds >= right] = lower_bounds[lower_bounds >= right]
    solution[upper_bounds <= left] = upper_bounds[upper_bounds <= left]
    return x + solution


def get_item(x, k, cond=None):
    if cond is not None:
        k = k.masked_fill(~cond, 0)
    result = x.gather(1, k.unsqueeze(dim=-1)).squeeze(1)
    if cond is not None:
        return result[cond]
    return result


def _project_onto_knapsack_constraint_batch(xs):
    n, d = xs.size()

    lower_bounds = -xs
    upper_bounds = 1 - xs

    total_weight = 1 - xs.sum(dim=1)
    lower_sorted, lower_sorted_indices = torch.sort(lower_bounds, dim=1)
    upper_sorted, upper_sorted_indices = torch.sort(upper_bounds, dim=1)
    tight_sum = torch.sum(lower_bounds, dim=1)

    slack_weight = xs.new_zeros((n,), dtype=torch.float)
    level = xs.new_zeros((n,), dtype=torch.long)
    k = xs.new_zeros((n,), dtype=torch.long)
    l = xs.new_zeros((n,), dtype=torch.long)
    left = xs.new_full((n,), -INF, dtype=torch.float)
    right = xs.new_full((n,), -INF, dtype=torch.float)
    not_found = xs.new_ones((n,), dtype=torch.bool)
    tau = xs.new_zeros((n,), dtype=torch.float)
    value_lower = xs.new_zeros((n,), dtype=torch.float)
    value_upper = xs.new_zeros((n,), dtype=torch.float)
    index_lower = xs.new_zeros((n,), dtype=torch.long)
    index_upper = xs.new_zeros((n,), dtype=torch.long)
    to_search = xs.new_ones((n,), dtype=torch.bool)

    # for _ in range(2 * d):
    while to_search.any():

        mask = to_search & not_found

        cond = mask & (level != 0)
        tau[cond] = (total_weight[cond] - tight_sum[cond]) / slack_weight[cond]

        cond = mask & (k < d)
        value_lower[cond] = get_item(lower_sorted, k, cond)
        value_lower[mask & (k >= d)] = INF
        index_lower[cond] = get_item(lower_sorted_indices, k, cond)

        cond = mask & (l < d)
        value_upper[cond] = get_item(upper_sorted, l, cond)
        value_upper[mask & (l >= d)] = INF
        index_upper[cond] = get_item(upper_sorted_indices, l, cond)

        left[mask] = right[mask]
        right[mask] = torch.min(value_lower, value_upper)[mask]

        not_found &= torch.where(
            level == 0, (left > tau) | (tau > right), total_weight != tight_sum
        )
        if not not_found.any():
            break

        cond = mask & (value_lower < value_upper)
        tight_sum[cond] -= get_item(lower_bounds, index_lower)[cond]
        slack_weight[cond] += 1.0
        level[cond] += 1
        k[cond] += 1

        cond = mask & (value_lower >= value_upper)
        tight_sum[cond] += get_item(upper_bounds, index_upper)[cond]
        slack_weight[cond] -= 1.0
        level[cond] -= 1
        l[cond] += 1

        to_search &= (k < d) | (l < d)

    solution = tau[:, None].repeat(1, d)
    left[not_found] = right[not_found]
    right[not_found] = INF
    left, right = left[:, None], right[:, None]

    solution[lower_bounds >= right] = lower_bounds[lower_bounds >= right]
    solution[upper_bounds <= left] = upper_bounds[upper_bounds <= left]

    return xs + solution


def project_onto_knapsack_constraint_batch(xs, mask=None, padding=0.0):
    if mask is None:
        return _project_onto_knapsack_constraint_batch(xs)
    assert len(mask.size()) == 1
    assert xs.size(0) == len(mask)
    _, d = xs.size()
    targets = torch.masked_select(xs, mask[:, None]).view(-1, d)
    projected = _project_onto_knapsack_constraint_batch(targets)
    output = torch.ones_like(xs) * padding
    output[mask] = projected
    return output
