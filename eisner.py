
import torch

# ROOT this is a pen
scores = torch.tensor([
    [0., 0., 0., 0., 0., 0.],  # ROOT
    [0., 0., 1., 0., 0., 0.],  # this
    [1., 0., 0., 0., 0., 0.],  # is
    [0., 0., 0., 0., 0., 1.],  # a
    [0., 0., 0., 0., 0., 1.],  # great
    [0., 0., 1., 0., 0., 0.],  # pen
], dtype=torch.float)

L, R = 0, 1

def eisner_parsing(scores):
    N = len(scores)
    chart = scores.new_zeros((N, N, 2, 2), dtype=torch.float)
    trace = scores.new_zeros((N, N, 2, 2), dtype=torch.int)
    for k in range(1, N):
        for s in range(N):
            t = s + k
            if t >= N: continue

            val, ind = torch.max(chart[s, s:t, R, 1] + chart[s+1:t+1, t, L, 1], dim=0)
            chart[s, t, L, 0] = val + scores[s, t]
            chart[s, t, R, 0] = val + scores[t, s]
            trace[s, t, :, 0] = s + ind

            val, ind = torch.max(chart[s, s:t, L, 1] + chart[s:t, t, L, 0], dim=0)
            chart[s, t, L, 1] = val
            trace[s, t, L, 1] = s + ind
            val, ind = torch.max(chart[s, s+1:t+1, R, 0] + chart[s+1:t+1, t, R, 1], dim=0)
            chart[s, t, R, 1] = val
            trace[s, t, R, 1] = s + 1 + ind

    result = scores.new_zeros(N, dtype=torch.int)
    def backtrack(s, t, direction, complete):
        if s == t: return
        k = trace[s, t, direction, complete]
        if complete:
            if direction == R:
                backtrack(s, k, R, 0); backtrack(k, t, R, 1)
            else:
                backtrack(s, k, L, 1); backtrack(k, t, L, 0)
        else:
            if direction == R:
                result[t] = s
            else:
                result[s] = t
            backtrack(s, k, R, 1); backtrack(k + 1, t, L, 0)
    backtrack(0, N-1, R, 1)
    return result

print(eisner_parsing(scores))
