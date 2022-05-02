import models_resnet
import models_AASIST
import prepare
import torch
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from lime import lime_image
import numpy as np

from skimage.color import gray2rgb
from skimage.color import rgb2gray

num_samples = 1000
features = 50000
weights_1d_resnet = "/time_frame_149_1d-resnet_Loss_0.0003_dEER_0.81%_eEER_0.76%.pth"
weights_2d_resnet = "/cqt_23_2d-resnet_Loss_0.004_dEER_0.77%_eEER_0.89%.pth"
weights_AASIST = "/AASIST_weights.pth"


def batch_predict_1d_resnet(sample):
    model, device = prepare.load_1d_resnet(weights_1d_resnet)

    sample = rgb2gray(sample)
    sample = torch.tensor(sample, dtype=torch.float32)

    test_loader = DataLoader(sample, batch_size=100, shuffle=False, num_workers=6)
    for test_batch in test_loader:
        test_batch = torch.unsqueeze(test_batch, 1)
        test_batch = torch.squeeze(test_batch,3)
        test_batch = test_batch.to(device)
        infer = model(test_batch)

    return(infer.detach().cpu().numpy())


def explain_1d_resnet(file):

    file3d = gray2rgb(np.expand_dims(file, 1))

    explainer = lime_image.LimeImageExplainer(kernel_width=0.5, random_state=40)
    explanation = explainer.explain_instance(file3d,
                                             batch_predict_1d_resnet,  # classification function
                                             batch_size=100,
                                             num_features=features,
                                             num_samples=num_samples)  # number of images that will be sent to classification function

    torch.cuda.empty_cache()

    explanation = explanation.image.T[0][0]
    explanation[explanation > explanation] = 0

    return abs(explanation)

def batch_predict_2d_resnet(cqt):
    model, device = prepare.load_2d_resnet(weights_2d_resnet)
    sample = rgb2gray(cqt)

    test_loader = DataLoader(sample, batch_size=100, shuffle=False, num_workers=3)
    for test_batch in test_loader:
        test_batch = test_batch.to(device=device, dtype=torch.float)
        test_batch = torch.unsqueeze(test_batch, 1)
        infer = model(test_batch)

    return infer.detach().cpu().numpy()


def explain_2d_resnet(cqt):


    explainer = lime_image.LimeImageExplainer(kernel_width=0.5, random_state=40)
    explanation = explainer.explain_instance(cqt,
                                             batch_predict_2d_resnet,  # classification function
                                             batch_size=100,
                                             num_features=features,
                                             num_samples=num_samples)  # number of images that will be sent to classification function

    torch.cuda.empty_cache()
    return explanation.image.T[0].T

def batch_predict_aasist(file):
    model, device = prepare.load_aasist(weights_AASIST)
    sample = rgb2gray(file)
    sample = np.squeeze(sample)
    test_loader = DataLoader(sample, batch_size=10, shuffle=False, num_workers=4)
    for test_batch in test_loader:
        test_batch = test_batch.to(device=device, dtype=torch.float)
        infer = model(test_batch)

    return infer.detach().cpu().numpy()

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def explain_aasist(file):
    X_pad = pad(file, 64600)
    x_inp = Tensor(X_pad)
    file3d = gray2rgb(np.expand_dims(x_inp, 1))
    explainer = lime_image.LimeImageExplainer(kernel_width=0.5, random_state=40)
    explanation = explainer.explain_instance(file3d,
                                             batch_predict_aasist,  # classification function
                                             batch_size=10,
                                             num_features=features,
                                             num_samples=num_samples)  # number of images that will be sent to classification function

    torch.cuda.empty_cache()
    explanation = explanation.image.T[0][0]
    explanation[explanation > explanation] = 0

    return abs(explanation)
