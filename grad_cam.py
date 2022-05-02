import torch

from captum.attr import GuidedGradCam
import numpy as np


def explain_1d_resnet(loader, model, device, state):

    for test_batch in loader:
        test_batch = torch.squeeze(test_batch, 2)
        test_batch = test_batch.to(device)

    guided_gc = GuidedGradCam(model, model.RSM4)
    attributions_gc = guided_gc.attribute(test_batch, state)

    torch.cuda.empty_cache()

    return attributions_gc.detach().cpu().numpy()[0][0]


def explain_2d_resnet(loader, model, device, state):

    for test_batch in loader:
        test_batch = torch.unsqueeze(test_batch, 0)
        test_batch = test_batch.to(device)

    guided_gc = GuidedGradCam(model, model.RSM4)
    attributions_gc = guided_gc.attribute(test_batch, state)

    torch.cuda.empty_cache()

    return attributions_gc.detach().cpu().numpy()[0][0]

