import torch
from OfflineOnline.LMU.flattened_layer import FlatLinear

# tests for 1 to 1 averaged correspondence
rows = 5
columns = 7
outputs = 4
batch = 3
L = FlatLinear(rows, columns, outputs, 'cpu') # type:ignore
t = torch.rand(columns*batch, dtype=torch.float32)
expected_output = t.reshape(batch, columns)
t = t.repeat(rows, 1).reshape(rows, batch, columns).transpose(dim0=0, dim1=1)
if outputs == columns:
    print("error amount is:")
    print(expected_output - L(t))
else:
    print(t)
    print(L(t))

