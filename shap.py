import prepare

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization

import torch
import numpy

num_samples = 60


def explain_1d_resnet(loader, model, device, state):
    for test_batch in loader:
        test_batch = torch.squeeze(test_batch, 2)
        test_batch = test_batch.to(device)

    gradient_shap = GradientShap(model)
    bsline = torch.zeros(test_batch.shape).to(device)
    target = torch.tensor(state).to(device)

    attributions_gs = gradient_shap.attribute(test_batch,
                                              n_samples=num_samples,
                                              baselines=bsline,
                                              target=target)
    torch.cuda.empty_cache()

    return attributions_gs.detach().cpu().numpy()[0][0]


def explain_2d_resnet(loader, model, device, state):

    for test_batch in loader:
        test_batch = torch.unsqueeze(test_batch, 0)
        test_batch = test_batch.to(device)

    gradient_shap = GradientShap(model)
    bsline = torch.zeros(test_batch.shape).to(device)
    target = torch.tensor(state).to(device)

    attributions_gs = gradient_shap.attribute(test_batch,
                                              n_samples=num_samples,
                                              baselines=bsline,
                                              target=target)
    torch.cuda.empty_cache()

    return attributions_gs.detach().cpu().numpy()[0][0]


def explain_AASIST(loader, model, device, state):

    if state == 1:
        state = 0
    if state == 0:
        state = 1
    for test_batch in loader:
        test_batch = test_batch.to(device)

    gradient_shap = GradientShap(model)
    bsline = torch.zeros(test_batch.shape).to(device)
    target = torch.tensor(state).to(device)

    attributions_gs = gradient_shap.attribute(test_batch,
                                              n_samples=8,
                                              baselines=bsline,
                                              target=target)
    torch.cuda.empty_cache()

    return attributions_gs.detach().cpu().numpy()[0]


def bulk_explain_1d_resnet(loader, model, device, state):
    for test_batch in loader:
        test_batch = torch.unsqueeze(test_batch, 1)
        test_batch = test_batch.to(device)

    gradient_shap = GradientShap(model)
    bsline = torch.zeros(test_batch.shape).to(device)
    target = torch.tensor(state).to(device)

    attributions_gs = gradient_shap.attribute(test_batch,
                                              n_samples=50,
                                              baselines=bsline,
                                              target=target)
    torch.cuda.empty_cache()

    return attributions_gs.detach().cpu().numpy()[0][0]


def bulk_explain_2d_resnet(loader, model, device, state):
    for test_batch in loader:
        test_batch = torch.unsqueeze(test_batch, 1)
        test_batch = test_batch.to(device)

    gradient_shap = GradientShap(model)
    bsline = torch.zeros(test_batch.shape).to(device)
    target = torch.tensor(state).to(device)

    attributions_gs = gradient_shap.attribute(test_batch,
                                              n_samples=50,
                                              baselines=bsline,
                                              target=target)
    torch.cuda.empty_cache()

    return attributions_gs.detach().cpu().numpy()[0][0]
