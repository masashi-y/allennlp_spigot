
import torch

INF = float('inf')


def project_onto_knapsack_constraint(x, weights=None):
    """
    solves the following problem:
     \min_v  || v ||^2 s.t.
      sum_i v_i = 1 - sum_i x_i,
      - x_i <= v_i <= 1 - x_i  (i = 1, ..., n)
    and returns x + v, which is projection of x onto simplex,
    satisfying sum_i x_i + v_i = 1 and 0 <= x_i + v_i <= 1 (i = 1, ..., n).
    """
    d, = x.size()
    lower_bounds = - x
    upper_bounds = 1 - x
    if weights is None:
        weights = x.new_ones((d,), dtype=torch.float)
    total_weight = 1 - x.sum()
    lower_sorted, lower_sorted_indices = torch.sort(lower_bounds)
    upper_sorted, upper_sorted_indices = torch.sort(upper_bounds)
    tight_sum = torch.sum(lower_bounds * weights)
    lower_sorted = torch.cat([lower_sorted, x.new_tensor([INF])])
    upper_sorted = torch.cat([upper_sorted, x.new_tensor([INF])])

    slack_weight = 0.
    level = k = l = 0
    left = right = - INF
    found = False

    while k < d or l < d:
        if level != 0:
            tau = (total_weight - tight_sum) / slack_weight

        left, right = right, min(lower_sorted[k], upper_sorted[l])

        if slack_weight == 0 and total_weight == tight_sum or \
                slack_weight != 0 and left <= tau <= right:
            found = True
            break

        if lower_sorted[k] < upper_sorted[l]:
            index = lower_sorted_indices[k]
            tight_sum -= lower_bounds[index] * weights[index]
            slack_weight += weights[index]
            level += 1
            k += 1
        else:
            index = upper_sorted_indices[l]
            tight_sum += upper_bounds[index] * weights[index]
            slack_weight -= weights[index]
            level -= 1
            l += 1

    solution = x.new_full((d,), tau, dtype=torch.float)
    if not found:
        left, right = right, INF
    solution[lower_bounds >= right] = lower_bounds[lower_bounds >= right]
    solution[upper_bounds <= left] = upper_bounds[upper_bounds <= left]
    return x + solution


def get_item(x, k):
    return x.gather(1, k).squeeze(1)


def project_onto_knapsack_constraint_batch(xs):
    n, d = xs.size()
    inf_tensor = xs.new_full((n, 1), INF)
    lower_bounds = torch.cat([- xs, inf_tensor], axis=1)
    upper_bounds = torch.cat([1 - xs, inf_tensor], axis=1)
    weights = xs.new_ones((n, d), dtype=torch.float)
    # weights[:, -1] = 0.
    total_weight = 1 - xs.sum(dim=1)
    lower_sorted, lower_sorted_indices = torch.sort(lower_bounds, dim=1)
    upper_sorted, upper_sorted_indices = torch.sort(upper_bounds, dim=1)
    tight_sum = torch.sum(lower_bounds[:, :-1] * weights, dim=1)

    slack_weight = xs.new_zeros((n,), dtype=torch.float)
    level = xs.new_zeros((n,), dtype=torch.long)
    k = xs.new_zeros((n, 1), dtype=torch.long)
    l = xs.new_zeros((n, 1), dtype=torch.long)
    left = xs.new_full((n,), INF, dtype=torch.float)
    right = xs.new_full((n,), INF, dtype=torch.float)
    not_found = xs.new_ones((n,), dtype=torch.bool)
    tau = xs.new_zeros((n,), dtype=torch.float)

    for _ in range(2 * d):
        cond = not_found.logical_and(level != 0)
        tau[cond] = (total_weight[cond] - tight_sum[cond]) / slack_weight[cond]

        a = get_item(lower_sorted, k)
        b = get_item(upper_sorted, l)
        left[not_found] = right[not_found]
        right[not_found] = torch.min(a, b)[not_found]

        not_found &= (level == 0).logical_and(total_weight == tight_sum).logical_not_()
        not_found &= (level != 0).logical_and(left <= tau).logical_and(tau <= right).logical_not_()
        if not not_found.any(): break

        index_a = lower_sorted_indices.gather(1, k)
        index_b = upper_sorted_indices.gather(1, l)
        weights_a = get_item(weights, index_a)
        weights_b = get_item(weights, index_b)

        cond = not_found.logical_and(a < b)
        tight_sum[cond] -= get_item(lower_bounds, index_a)[cond] * weights_a[cond]
        slack_weight[cond] += weights_a[cond]
        level[cond] += 1
        k[cond] += 1

        cond = not_found.logical_and(a >= b)
        tight_sum[cond] += get_item(upper_bounds, index_b)[cond] * weights_b[cond]
        slack_weight[cond] -= weights_b[cond]
        level[cond] -= 1
        l[cond] += 1

    solution = tau[:, None].repeat(1, d)
    left[not_found] = right[not_found]
    right[not_found] = INF
    lower_bounds = lower_bounds[:, :-1]
    upper_bounds = upper_bounds[:, :-1]
    left, right = left[:, None], right[:, None]
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    solution[lower_bounds >= right] = lower_bounds[lower_bounds >= right]
    solution[upper_bounds <= left] = upper_bounds[upper_bounds <= left]
    return xs + solution


