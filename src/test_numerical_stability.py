import argparse

import numpy as np
import torch
import torch.nn as nn

parse = argparse.ArgumentParser()

parse.add_argument(
    "-d",
    "--device",
    default="cpu",
    type=str,
)
parse.add_argument(
    "-t",
    "--type",
    "--torch.float64",
    default="torch.float64",
    type=str,
)

parse.add_argument(
    "--discrete_targets",
    action="store_true",
)

parse.add_argument(
    "--range",
    default=10,
    type=float,
)

args = parse.parse_args()


device = args.device
dtype = args.type
print(f"device={device}")
print(f"dtype={dtype}")

torch.manual_seed(0)

a = torch.tensor(
    (torch.rand(10000000, dtype=float) * 2.0 * args.range - args.range).tolist(),
    dtype=eval(dtype),
).tolist()
b = torch.rand(10000000, dtype=eval(dtype))

if args.discrete_targets:
    b = b > 0.5

b = b.tolist()

print()
print("input:")
print(f"max(i)={max(a)}")
print(f"min(i)={min(a)}")
print()
print("targets:")
print(f"args.discrete_targets={args.discrete_targets}")

gt_loss = nn.BCEWithLogitsLoss().to("cpu")
gt_i = torch.tensor(a, requires_grad=True, device="cpu", dtype=torch.float64)
gt_t = torch.tensor(b, requires_grad=True, device="cpu", dtype=torch.float64)
gt_output = gt_loss(gt_i, gt_t)
gt_output.backward()
avg_gt_value = np.average(abs(gt_i.grad))

print()
print("nn.BCEWithLogitsLoss()")
loss = nn.BCEWithLogitsLoss().to(device)
i = torch.tensor(a, requires_grad=True, device=device, dtype=eval(dtype))
t = torch.tensor(b, requires_grad=True, device=device, dtype=eval(dtype))
output = loss(i, t)
output.backward()
i_grad = i.grad.to("cpu")
avg_error = np.average(abs(i_grad - gt_i.grad))
max_error = torch.max(abs(i_grad - gt_i.grad))
print(f"average gradient value = {avg_gt_value}")
print(f"average absolute error = {avg_error}")
print(f"maximum absolute error = {max_error}")
print()


print("nn.BCEWithLogitsLoss(pos_weight=[1.0])")
loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0)).to(device)
i = torch.tensor(a, requires_grad=True, device=device, dtype=eval(dtype))
t = torch.tensor(b, requires_grad=True, device=device, dtype=eval(dtype))
output = loss(i, t)
output.backward()
i_grad = i.grad.to("cpu")
avg_error2 = np.average(abs(i_grad - gt_i.grad))
max_error2 = torch.max(abs(i_grad - gt_i.grad))
print(f"average gradient value = {avg_gt_value}")
print(
    f"average absolute error = {avg_error2} (relative diff {(avg_error2 - avg_error)/(avg_error)})"
)
print(
    f"maximum absolute error = {max_error2} (relative diff {(max_error2 - max_error)/(max_error)})"
)
