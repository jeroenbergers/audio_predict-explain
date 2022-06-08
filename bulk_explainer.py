import bulk_prepare
import prepare
import SHAP
import grad_cam
import main
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import display
import LIME


def explain_1d_shap(test_set, resnet1d_model, device, attack_type):
    test_loader = DataLoader(test_set[0][0], batch_size=1, shuffle=False, num_workers=1)
    x = SHAP.bulk_explain_1d_resnet(test_loader, resnet1d_model, device, 1)
    mask_list = x
    i = 0

    if attack_type is not None:
        while i < len(test_set):
            if test_set[i][2] == torch.tensor(attack_type):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = SHAP.bulk_explain_1d_resnet(test_loader, resnet1d_model, device, 1)
                mask_list = np.vstack([mask_list, x])
                if mask_list.T[0].shape[0] == 300:
                    i = len(test_set)
            i = i + 1
    else:
        while i < len(test_set):
            if test_set[i][1] == torch.tensor(1):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = SHAP.bulk_explain_1d_resnet(test_loader, resnet1d_model, device, 1)
                mask_list = np.vstack([mask_list, x])
            i = i + 1

    return mask_list[1:]


def explain_1d_gc(test_set, resnet1d_model, device, attack_type):
    test_loader = DataLoader(test_set[0][0], batch_size=1, shuffle=False, num_workers=1)
    x = grad_cam.bulk_explain_1d_resnet(test_loader, resnet1d_model, device, 1)
    mask_list = x
    i = 0

    if attack_type is not None:
        while i < len(test_set):
            if test_set[i][2] == torch.tensor(attack_type):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.bulk_explain_1d_resnet(test_loader, resnet1d_model, device, 1)
                mask_list = np.vstack([mask_list, x])
                if mask_list.T[0].shape[0] == 100:
                    i = len(test_set)
            i = i + 1
    else:
        while i < len(test_set):
            if test_set[i][1] == torch.tensor(1):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.bulk_explain_1d_resnet(test_loader, resnet1d_model, device, 1)
                mask_list = np.vstack([mask_list, x])
            i = i + 1

    return mask_list[1:]

def explain_1d_guided(test_set, resnet1d_model, device, attack_type):
    test_loader = DataLoader(test_set[0][0], batch_size=1, shuffle=False, num_workers=1)
    x = grad_cam.bulk_explain_1d_guided(test_loader, resnet1d_model, device, 1)
    mask_list = x
    i = 0

    if attack_type is not None:
        while i < len(test_set):
            if test_set[i][2] == torch.tensor(attack_type):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.bulk_explain_1d_guided(test_loader, resnet1d_model, device, 1)
                mask_list = np.vstack([mask_list, x])
                if mask_list.T[0].shape[0] == 100:
                    i = len(test_set)
            i = i + 1
    else:
        while i < len(test_set):
            if test_set[i][1] == torch.tensor(1):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.bulk_explain_1d_guided(test_loader, resnet1d_model, device, 1)
                mask_list = np.vstack([mask_list, x])
            i = i + 1

    return mask_list[1:]



def explain_1d_lime(test_set, attack_type):
    x = LIME.explain_1d_resnet(test_set[0][0][0])
    mask_list = x
    i = 0

    if attack_type is not None:
        while i < len(test_set):
            if test_set[i][2] == torch.tensor(attack_type):
                x = LIME.explain_1d_resnet(test_set[0][0][0])
                mask_list = np.vstack([mask_list, x])
                if mask_list.T[0].shape[0] == 100:
                    i = len(test_set)
            i = i + 1
    else:
        while i < len(test_set):
            if test_set[i][1] == torch.tensor(1):
                x = LIME.explain_1d_resnet(test_set[0][0][0])
                mask_list = np.vstack([mask_list, x])
            i = i + 1

    return mask_list[1:]


def explain_2d_shap(test_set, resnet1d_model, device, attack_type):
    test_loader = DataLoader(test_set[0][0], batch_size=1, shuffle=False, num_workers=1)
    x = SHAP.bulk_explain_2d_resnet(test_loader, resnet1d_model, device, 1)
    x = np.expand_dims(x, axis=0)
    mask_list = x
    i = 0

    if attack_type is not None:
        while i < len(test_set):
            if test_set[i][2] == torch.tensor(attack_type):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = SHAP.bulk_explain_2d_resnet(test_loader, resnet1d_model, device, 1)
                x = np.expand_dims(x, axis=0)
                mask_list = np.vstack([mask_list, x])
                if mask_list.shape[0] >= 300:
                    i = len(test_set)
            i = i + 1
    else:
        while i < len(test_set):
            if test_set[i][1] == torch.tensor(1):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = SHAP.bulk_explain_2d_resnet(test_loader, resnet1d_model, device, 1)
                x = np.expand_dims(x, axis=0)
                mask_list = np.vstack([mask_list, x])
            i = i + 1

    return mask_list[1:]


def explain_2d_guided(test_set, resnet2d_model, device, attack_type):
    test_loader = DataLoader(test_set[0][0], batch_size=1, shuffle=False, num_workers=1)
    x = grad_cam.explain_2d_resnet_guided(test_loader, resnet2d_model, device, 1)
    mask_list = np.expand_dims(x, axis=0)
    i = 0

    if attack_type is not None:
        while i < len(test_set):
            if test_set[i][2] == torch.tensor(attack_type):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.explain_2d_resnet_guided(test_loader, resnet2d_model, device, 1)
                x = np.expand_dims(x, axis=0)
                mask_list = np.vstack([mask_list, x])
                if mask_list.shape[0] >= 100:
                    i = len(test_set)
            i = i + 1
    else:
        while i < len(test_set):
            if test_set[i][1] == torch.tensor(1):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.explain_2d_resnet_guided(test_loader, resnet2d_model, device, 1)
                x = np.expand_dims(x, axis=0)
                mask_list = np.vstack([mask_list, x])
            i = i + 1

    return mask_list[1:]

def explain_2d_gc(test_set, resnet2d_model, device, attack_type):
    test_loader = DataLoader(test_set[0][0], batch_size=1, shuffle=False, num_workers=1)
    x = grad_cam.explain_2d_resnet(test_loader, resnet2d_model, device, 1)
    mask_list = np.expand_dims(x, axis=0)
    i = 0

    if attack_type is not None:
        while i < len(test_set):
            if test_set[i][2] == torch.tensor(attack_type):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.explain_2d_resnet(test_loader, resnet2d_model, device, 1)
                x = np.expand_dims(x, axis=0)
                mask_list = np.vstack([mask_list, x])
                if mask_list.shape[0] >= 100:
                    i = len(test_set)
            i = i + 1
    else:
        while i < len(test_set):
            if test_set[i][1] == torch.tensor(1):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.explain_2d_resnet(test_loader, resnet2d_model, device, 1)
                x = np.expand_dims(x, axis=0)
                mask_list = np.vstack([mask_list, x])
            i = i + 1

    return mask_list[1:]



def explain_aasist_shap(test_set, aasist_model, device, attack_type):
    test_loader = DataLoader(test_set[0][0], batch_size=1, shuffle=False, num_workers=1)
    x = SHAP.explain_AASIST(test_loader, aasist_model, device, 0)
    mask_list = x
    i = 0

    if attack_type is not None:
        while i < len(test_set):
            if test_set[i][2] == torch.tensor(attack_type):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = SHAP.explain_AASIST(test_loader, aasist_model, device, 0)
                mask_list = np.vstack([mask_list, x])
                if mask_list.shape[0] >= 200:
                    i = len(test_set)
            i = i + 1
    else:
        while i < len(test_set):
            if test_set[i][1] == torch.tensor(1):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = SHAP.explain_AASIST(test_loader, aasist_model, device, 0)
                mask_list = np.vstack([mask_list, x])
            i = i + 1

    return mask_list[1:]


def explain_aasist_guided(test_set, aasist_model, device, attack_type):
    test_loader = DataLoader(test_set[0][0], batch_size=1, shuffle=False, num_workers=1)
    x = grad_cam.explain_AASIST_guided(test_loader, aasist_model, device, 0)
    mask_list = x
    i = 0

    if attack_type is not None:
        while i < len(test_set):
            if test_set[i][2] == torch.tensor(attack_type):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.explain_AASIST_guided(test_loader, aasist_model, device, 0)
                mask_list = np.vstack([mask_list, x])
                if mask_list.shape[0] >= 100:
                    i = len(test_set)
            i = i + 1
    else:
        while i < len(test_set):
            if test_set[i][1] == torch.tensor(1):
                test_loader = DataLoader(test_set[i][0], batch_size=1, shuffle=False, num_workers=1)
                x = grad_cam.explain_AASIST_guided(test_loader, aasist_model, device, 0)
                mask_list = np.vstack([mask_list, x])
            i = i + 1

    return mask_list[1:]


def resnet_1d_shap(protocol_file_path, database_path, data_type,  attack_type=None):
    resnet1d_model, device = prepare.load_1d_resnet(main.weights_1d_resnet)
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_1d_shap(test_set,  resnet1d_model, device, attack_type)
    display.bulk_1d(mask_list)
    return mask_list


def resnet_1d_gc(protocol_file_path, database_path, data_type, attack_type=None):
    resnet1d_model, device = prepare.load_1d_resnet(main.weights_1d_resnet)
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_1d_gc(test_set,  resnet1d_model, device, attack_type)
    display.bulk_1d(mask_list)
    return mask_list


def resnet_1d_lime(protocol_file_path, database_path, data_type, attack_type=None):
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_1d_lime(test_set,  attack_type)
    display.bulk_1d(mask_list)
    return mask_list


def resnet_1d_guided(protocol_file_path, database_path, data_type, attack_type=None):
    resnet1d_model, device = prepare.load_1d_resnet(main.weights_1d_resnet)
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_1d_guided(test_set,  resnet1d_model, device, attack_type)
    display.bulk_1d(mask_list)
    return mask_list


def resnet_2d_shap(protocol_file_path, database_path, data_type,  attack_type=None):
    resnet2d_model, device = prepare.load_2d_resnet(main.weights_2d_resnet)
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_2d_shap(test_set,  resnet2d_model, device, attack_type)
    display.bulk_2d(mask_list)
    return mask_list


def resnet_2d_gc(protocol_file_path, database_path, data_type, attack_type=None):
    resnet2d_model, device = prepare.load_2d_resnet(main.weights_2d_resnet)
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_2d_gc(test_set,  resnet2d_model, device, attack_type)
    display.bulk_2d(mask_list)
    return mask_list


def resnet_2d_lime(protocol_file_path, database_path, data_type, attack_type=None):
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_2d_lime(test_set,  attack_type)
    display.bulk_2d(mask_list)
    return mask_list


def resnet_2d_guided(protocol_file_path, database_path, data_type, attack_type=None):
    resnet2d_model, device = prepare.load_2d_resnet(main.weights_2d_resnet)
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_2d_guided(test_set,  resnet2d_model, device, attack_type)
    display.bulk_2d(mask_list)
    return mask_list


def AASIST_1d_shap(protocol_file_path, database_path, data_type,  attack_type=None):
    aasist_model, device = prepare.load_aasist(main.weights_AASIST)
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_aasist(test_set,  aasist_model, device, attack_type)
    display.bulk_1d(mask_list)
    return mask_list


def AASIST_1d_guided(protocol_file_path, database_path, data_type,  attack_type=None):
    aasist_model, device = prepare.load_aasist(main.weights_AASIST)
    test_set = bulk_prepare.resnet(protocol_file_path, database_path, data_type)
    mask_list = explain_aasist_guided(test_set,  aasist_model, device, attack_type)
    display.bulk_1d(mask_list)
    return mask_list

