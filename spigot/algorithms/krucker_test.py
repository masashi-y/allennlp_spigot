
import torch
import subprocess
from spigot.algorithms.krucker import (
        project_onto_knapsack_constraint,
        project_onto_knapsack_constraint_batch
        )

def test():
    x = torch.randn((6,)).float()
    input_text = str(x.size(0)) + ' ' + ' '.join(f'{float(v):.4}' for v in x)
    proc = subprocess.run("./a.out", shell=True, input=input_text, stdout=subprocess.PIPE, text=True)
    cpp = [float(v) for v in proc.stdout.strip().split(' ')]
    cpp_res = torch.tensor(cpp, dtype=torch.float)
    cpp_dot = cpp_res.dot(x)
    print('input:', '[' + ', '.join(f'{v:.4}' for v in x) + ']')
    print(f'c++(sum: {sum(cpp):.4}, dot:{cpp_dot:.4}):', cpp)
    py_res = project_onto_knapsack_constraint(x)
    py = [float(v) for v in py_res]
    py_dot = py_res.dot(x).item()
    py_sum = py_res.sum().item()
    print(f'py (sum: {py_sum:.4}, dot:{py_dot:.4}):', py)
    print(torch.allclose(py_res, cpp_res, ))


if __name__ == '__main__':
    for _ in range(10):
        test()
    print(project_onto_knapsack_constraint_batch(torch.tensor([[-10, -10000, 0], [-10, -10000, -10]]).float()))

    for i in range(10000):
        x = torch.randint(0, 100, (5,)).float()
        x[2:4] = torch.randint(-1000, 0, (2,)).float()
        res0 = project_onto_knapsack_constraint(x)
        weights = torch.ones((9,), dtype=torch.float)
        weights[5:] = 0.
        x = torch.cat([x, torch.zeros((4,), dtype=torch.float)])
        res1 = project_onto_knapsack_constraint(x, weights)[:5]
        if not torch.allclose(res0, res1):
            raise 'aaaaaaaaaaa'
        else:
            print(x)
            print(res0)
            print()
