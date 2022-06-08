import torch
import models_resnet
import models_AASIST
from importlib import import_module
from torch import Tensor

import numpy as np
import librosa
from scipy import signal

from torch.utils.data.dataloader import DataLoader


def load_1d_resnet(weights_1d_resnet):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models_resnet.SSDNet1D()
    check_point = torch.load("./trained_models" + weights_1d_resnet, map_location=device)
    model.load_state_dict(check_point['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device


def dataloader_1d(sample):
    sample = torch.tensor(sample, dtype=torch.float32)
    sample = torch.unsqueeze(sample, 0)
    sample = torch.unsqueeze(sample, 0)
    test_loader = DataLoader(sample, batch_size=1, shuffle=False, num_workers=4)

    return test_loader


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def load_aasist(weights_AASIST):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module = import_module("models_AASIST")
    _model = getattr(module, "Model")
    model = _model(
        {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]})\
        .to(device)

    check_point = torch.load("./trained_models" + weights_AASIST, map_location=device)
    model.load_state_dict(check_point)
    model.eval()
    return model, device


def dataloader_aasist(sample):
    x_inp = Tensor(sample)
    sample = torch.unsqueeze(x_inp, 0)
    data_loader = DataLoader(sample, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    return data_loader


def create_cqt(x, fs):
    # duration = 1
    # len_sample = int(duration * fs)
    # if len(x) < len_sample:
    # x = np.tile(x, int(len_sample // len(x)) + 1)
    # x = x[0: int(len_sample - 256)]

    x = signal.lfilter([1, -0.97], [1], x)
    x_cqt = librosa.cqt(x, sr=fs, hop_length=256, n_bins=432, bins_per_octave=48, window='hann', fmin=15)
    pow_cqt = np.square(np.abs(x_cqt))
    log_pow_cqt = 10 * np.log10(pow_cqt + 1e-30)
    return log_pow_cqt


def load_2d_resnet(weights_1d_resnet):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models_resnet.SSDNet2D()
    check_point = torch.load("./trained_models" + weights_1d_resnet, map_location=device)
    model.load_state_dict(check_point['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device


def dataloader_2d(sample):
    sample = torch.tensor(sample, dtype=torch.float32)
    sample = torch.unsqueeze(sample, 0)
    test_loader = DataLoader(sample, batch_size=1, shuffle=False, num_workers=4)
    return test_loader


def tanspose_audio_fragments(file, fs):
    # duration = 6
    # if len(file) < duration * fs:
    # file = np.tile(file, int((duration * fs) // len(file)) + 1)
    # file =  file[0: (int(duration * fs))]

    new_sr = 16000
    sample = librosa.resample(file, orig_sr=fs, target_sr=new_sr, res_type='sinc_best')
    return sample, new_sr


def tanspose_and_pad_audio_fragments(file, fs):
    duration = 6

    if len(file) < duration * fs:
        file = np.tile(file, int((duration * fs) // len(file)) + 1)
    file = file[0: (int(duration * fs))]

    new_sr = 16000
    sample = librosa.resample(file, orig_sr=fs, target_sr=new_sr, res_type='sinc_best')
    return sample, new_sr


def create_and_pad_cqt(x, fs):
    duration = 1

    len_sample = int(duration * fs)
    if len(x) < len_sample:
        x = np.tile(x, int(len_sample // len(x)) + 1)

    x = x[0: int(len_sample - 256)]

    x = signal.lfilter([1, -0.97], [1], x)
    x_cqt = librosa.cqt(x, sr=fs, hop_length=256, n_bins=432, bins_per_octave=48, window='hann', fmin=15)
    pow_cqt = np.square(np.abs(x_cqt))
    log_pow_cqt = 10 * np.log10(pow_cqt + 1e-30)
    return log_pow_cqt
