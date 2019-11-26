import math
import time
import torch
import torch.nn as nn
import numpy as np

if __name__ == "__main__":
    a = np.load("/checkpoint/cinjon/spaceofmotion/bsn/test_reps_random_amdim.npy")
    a = a.reshape((a.shape[0], -1))
    a = a - np.mean(a, 0, keepdims=True)
    a = torch.from_numpy(a.T).half().to(0)
    print(a.size())
    b = np.load("/checkpoint/cinjon/spaceofmotion/bsn/test_reps_random_amdim.npy")
    b = b.reshape((b.shape[0], -1))
    # b_shape = b.shape
    # third_size = int(b_shape[0] / 3)
    b = b - np.mean(b, 0, keepdims=True)
    # b1 = b[:third_size, :]
    # b2 = b[third_size:, :]
    # b = b1
    
    with torch.no_grad():
        b = torch.from_numpy(b).half().to(1)
        print(b.size())
        mat_mult = nn.Linear(in_features=b.shape[0], out_features=a.shape[0], bias=False)
        print(mat_mult.weight.size())
        mat_mult.weight.data = a
        mat_mult_gpu = nn.DataParallel(mat_mult, device_ids=[0, 1]).to('cuda:0')
        result = mat_mult_gpu(b.t())
        print(result.size())
        print(float(torch.norm(result.cpu())))
        # print(result.data.t())
