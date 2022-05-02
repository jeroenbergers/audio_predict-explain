import torch
import numpy
from torch import Tensor


def resnet1d(test_loader, model, device):

    for test_batch in test_loader:
        test_batch = torch.squeeze(test_batch, 2)
        test_batch = test_batch.to(device)
        infer = model(test_batch)
    return infer.detach().cpu()

def aasist(test_loader, model, device):
    for batch_x in test_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_out = model(batch_x)
        return batch_out.detach().cpu().T


def resnet2d(test_loader, model, device):

    for test_batch in test_loader:
        test_batch = torch.unsqueeze(test_batch, 0)
        test_batch = test_batch.to(device)
        infer = model(test_batch)

    return infer.detach().cpu()
