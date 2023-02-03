import torch

h = 64
w = 64

jacobian = torch.load("jacobian.pt")
jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)

coordinate_grid = torch.rand(1, 10, 64, 64, 2)
result = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))

# fail run
# result = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))

            