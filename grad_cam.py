import torch

from captum.attr import GuidedGradCam
from captum.attr import GuidedBackprop

import numpy as np
import warnings
warnings.filterwarnings("ignore")


def explain_1d_resnet(loader, model, device, state):

    for test_batch in loader:
        test_batch = torch.squeeze(test_batch, 2)
        test_batch = test_batch.to(device)

    guided_gc1 = GuidedGradCam(model, model.RSM1)
    guided_gc2 = GuidedGradCam(model, model.RSM2)
    guided_gc3 = GuidedGradCam(model, model.RSM3)
    guided_gc4 = GuidedGradCam(model, model.RSM4)

    attribution1 = guided_gc1.attribute(test_batch, state)
    attribution2 = guided_gc2.attribute(test_batch, state)
    attribution3 = guided_gc3.attribute(test_batch, state)
    attribution4 = guided_gc4.attribute(test_batch, state)

    attributions_gc = attribution1 + attribution2 + attribution3 + attribution4
    torch.cuda.empty_cache()

    return attributions_gc.detach().cpu().numpy()[0][0]


def explain_1d_resnet_guided(loader, model, device, state):

    for test_batch in loader:
        test_batch = torch.squeeze(test_batch, 2)
        test_batch = test_batch.to(device)

    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(test_batch, target=state)

    torch.cuda.empty_cache()

    return attribution.detach().cpu().numpy()[0][0]


def explain_2d_resnet(loader, model, device, state):

    for test_batch in loader:
        test_batch = torch.unsqueeze(test_batch, 0)
        test_batch = test_batch.to(device)

    guided_gc1 = GuidedGradCam(model, model.RSM1)
    guided_gc2 = GuidedGradCam(model, model.RSM2)
    guided_gc3 = GuidedGradCam(model, model.RSM3)
    guided_gc4 = GuidedGradCam(model, model.RSM4)

    attribution1 = guided_gc1.attribute(test_batch, state)
    attribution2 = guided_gc2.attribute(test_batch, state)
    attribution3 = guided_gc3.attribute(test_batch, state)
    attribution4 = guided_gc4.attribute(test_batch, state)

    attributions_gc = attribution1 + attribution2 + attribution3 + attribution4

    torch.cuda.empty_cache()

    return attributions_gc.detach().cpu().numpy()[0][0]


def explain_2d_resnet_guided(loader, model, device, state):

    for test_batch in loader:
        test_batch = torch.unsqueeze(test_batch, 0)
        test_batch = test_batch.to(device)

    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(test_batch, target=state)

    torch.cuda.empty_cache()

    return attribution.detach().cpu().numpy()[0][0]


def explain_AASIST_guided(loader, model, device, state):

    if state == 1:
        state = 0
    if state == 0:
        state = 1

    for test_batch in loader:
        test_batch = test_batch.to(device)

    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(test_batch, target=state)

    torch.cuda.empty_cache()
    print(attribution)
    print(attribution.shape)
    return attribution.detach().cpu().numpy()[0]


def bulk_explain_1d_resnet(loader, model, device, state):

    for test_batch in loader:
        test_batch = torch.unsqueeze(test_batch, 1)
        test_batch = test_batch.to(device)

    guided_gc1 = GuidedGradCam(model, model.RSM1)
    guided_gc2 = GuidedGradCam(model, model.RSM2)
    guided_gc3 = GuidedGradCam(model, model.RSM3)
    guided_gc4 = GuidedGradCam(model, model.RSM4)

    attribution1 = guided_gc1.attribute(test_batch, state)
    attribution2 = guided_gc2.attribute(test_batch, state)
    attribution3 = guided_gc3.attribute(test_batch, state)
    attribution4 = guided_gc4.attribute(test_batch, state)

    attributions_gc = attribution1 + attribution2 + attribution3 + attribution4
    torch.cuda.empty_cache()

    return attributions_gc.detach().cpu().numpy()[0][0]


def bulk_explain_1d_guided(loader, model, device, state):

    for test_batch in loader:
        test_batch = torch.unsqueeze(test_batch, 1)
        test_batch = test_batch.to(device)

    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(test_batch, target=state)

    torch.cuda.empty_cache()

    return attribution.detach().cpu().numpy()[0][0]
