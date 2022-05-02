import prepare
import predict
import display

import shap
import grad_cam
import LIME

#the initial model weights.
weights_1d_resnet = "/time_frame_149_1d-resnet_Loss_0.0003_dEER_0.81%_eEER_0.76%.pth"
weights_2d_resnet = "/cqt_23_2d-resnet_Loss_0.004_dEER_0.77%_eEER_0.89%.pth"
weights_AASIST = "/AASIST_weights.pth"

def predictions(sample, sr):
    #predictions input sample and sample rate, output predictions resnet1d and 2d and aasist

    # predict and prepare resnet1d, load model, predict outcome.
    resnet1d_model, device = prepare.load_1d_resnet(weights_1d_resnet)
    resnet1d_loader = prepare.dataloader_1d(sample)
    resnet1d_pred = predict.resnet1d(resnet1d_loader, resnet1d_model, device)

    #predict and prepare 2d resnet, create cqt, load model, predict outcome.
    cqt = prepare.create_cqt(sample, sr)
    resnet2d_model, device = prepare.load_2d_resnet(weights_2d_resnet)
    resnet2d_loader = prepare.dataloader_2d(cqt)
    resnet2d_pred = predict.resnet2d(resnet2d_loader, resnet2d_model, device)

    # predict and prepare aasist, load model, predict outcome.
    aasist_model, device = prepare.load_aasist(weights_AASIST)
    aasist_loader = prepare.dataloader_aasist(sample)
    aasist_pred = predict.aasist(aasist_loader, aasist_model, device)

    #display the predictions
    state1d, state2d, statea = display.predictions(resnet1d_pred, resnet2d_pred, aasist_pred)

    return resnet1d_model, resnet2d_model, aasist_model, cqt, device, state1d, state2d, statea


def explanations(resnet1d_model, resnet2d_model, aasist_model, cqt, device, sample, state1d, state2d, statea):

    #explain models, resnet 1d, resnet 2d and aasist, input the models and the sample, output 8 explanations:
    resnet1d_loader = prepare.dataloader_1d(sample)
    shap_1d = shap.explain_1d_resnet(resnet1d_loader, resnet1d_model, device, state1d)
    gc_1d = grad_cam.explain_1d_resnet(resnet1d_loader, resnet1d_model, device, state1d)
    lime_1d = LIME.explain_1d_resnet(sample)

    resnet2d_loader = prepare.dataloader_2d(cqt)
    shap_2d = shap.explain_2d_resnet(resnet2d_loader, resnet2d_model, device, state2d)
    gc_2d = grad_cam.explain_2d_resnet(resnet2d_loader, resnet2d_model, device, state2d)
    lime_2d = LIME.explain_2d_resnet(cqt)

    aasist_loader = prepare.dataloader_aasist(sample)
    shap_aasist = shap.explain_AASIST(aasist_loader, aasist_model, device, statea)
    lime_aasist = LIME.explain_aasist(sample)

    display.explanations(shap_1d, gc_1d, lime_1d, shap_2d, gc_2d, lime_2d, shap_aasist, lime_aasist, sample, cqt)
    return
