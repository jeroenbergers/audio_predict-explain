import prepare
import predict
import display
import soundfile as sf
import SHAP as shap
import grad_cam
import LIME

# the initial model weights.
weights_1d_resnet = "/time_frame_149_1d-resnet_Loss_0.0003_dEER_0.81%_eEER_0.76%.pth"
weights_2d_resnet = "/cqt_23_2d-resnet_Loss_0.004_dEER_0.77%_eEER_0.89%.pth"
weights_AASIST = "/AASIST_weights.pth"


def predictions(sample, sr):
    # predictions input sample and sample rate, output predictions resnet1d and 2d and aasist

    # predict and prepare resnet1d, load model, predict outcome.
    resnet1d_model, device = prepare.load_1d_resnet(weights_1d_resnet)
    resnet1d_loader = prepare.dataloader_1d(sample)
    resnet1d_pred = predict.resnet1d(resnet1d_loader, resnet1d_model, device)

    # predict and prepare 2d resnet, create cqt, load model, predict outcome.
    cqt = prepare.create_cqt(sample, sr)
    resnet2d_model, device = prepare.load_2d_resnet(weights_2d_resnet)
    resnet2d_loader = prepare.dataloader_2d(cqt)
    resnet2d_pred = predict.resnet2d(resnet2d_loader, resnet2d_model, device)

    # predict and prepare aasist, load model, predict outcome.
    aasist_model, device = prepare.load_aasist(weights_AASIST)
    aasist_loader = prepare.dataloader_aasist(sample)
    aasist_pred = predict.aasist(aasist_loader, aasist_model, device)

    # display the predictions
    state1d, state2d, statea = display.predictions(resnet1d_pred, resnet2d_pred, aasist_pred)

    return resnet1d_model, resnet2d_model, aasist_model, cqt, device, state1d, state2d, statea


def explanations(resnet1d_model, resnet2d_model, aasist_model, cqt, device, sample, state1d, state2d, statea):

    # explain models, resnet 1d, resnet 2d and aasist, input the models and the sample, output 8 explanations:
    resnet1d_loader = prepare.dataloader_1d(sample)
    shap_1d = shap.explain_1d_resnet(resnet1d_loader, resnet1d_model, device, state1d)
    gc_1d = grad_cam.explain_1d_resnet_guided(resnet1d_loader, resnet1d_model, device, state1d)
    lime_1d = LIME.explain_1d_resnet(sample)

    resnet2d_loader = prepare.dataloader_2d(cqt)
    shap_2d = shap.explain_2d_resnet(resnet2d_loader, resnet2d_model, device, state2d)
    gc_2d = grad_cam.explain_2d_resnet_guided(resnet2d_loader, resnet2d_model, device, state2d)
    lime_2d = LIME.explain_2d_resnet(cqt)

    aasist_loader = prepare.dataloader_aasist(sample)
    shap_aasist = shap.explain_AASIST(aasist_loader, aasist_model, device, statea)
    lime_aasist = LIME.explain_aasist(sample)

    # lime_1d = shap_1d
    # lime_2d = shap_2d
    # lime_aasist = shap_aasist
    display.explanations(shap_1d, gc_1d, lime_1d, shap_2d, gc_2d, lime_2d, shap_aasist, lime_aasist, sample, cqt)
    return


def explanations_1d_resnet(sample):
    resnet1d_model, device = prepare.load_1d_resnet(weights_1d_resnet)
    resnet1d_loader = prepare.dataloader_1d(sample)
    resnet1d_pred = predict.resnet1d(resnet1d_loader, resnet1d_model, device)
    state = display.resnet1d_score(resnet1d_pred)
    shap_1d = shap.explain_1d_resnet(resnet1d_loader, resnet1d_model, device, state)
    gc_1d = grad_cam.explain_1d_resnet(resnet1d_loader, resnet1d_model, device, state)
    lime_1d = LIME.explain_1d_resnet(sample)
    guided = grad_cam.explain_1d_resnet_guided(resnet1d_loader, resnet1d_model, device, state)
    print('1')
    display.resnet1d(lime_1d, sample)
    print('2')
    display.resnet1d(shap_1d, sample)
    print('3')
    display.resnet1d(guided, sample)
    print('4')
    display.resnet1d(gc_1d, sample)


def explanations_2d_resnet(sample, sr):
    cqt = prepare.create_cqt(sample, sr)
    resnet2d_model, device = prepare.load_2d_resnet(weights_2d_resnet)
    resnet2d_loader = prepare.dataloader_2d(cqt)
    resnet2d_pred = predict.resnet2d(resnet2d_loader, resnet2d_model, device)
    state = display.resnet2d_score(resnet2d_pred)
    shap_2d = shap.explain_2d_resnet(resnet2d_loader, resnet2d_model, device, state)
    gc_2d = grad_cam.explain_2d_resnet(resnet2d_loader, resnet2d_model, device, state)
    lime_2d = LIME.explain_2d_resnet(cqt)
    guided = grad_cam.explain_2d_resnet_guided(resnet2d_loader, resnet2d_model, device, state)
    print('1')
    display.resnet2d_lime(lime_2d, cqt)
    print('2')
    display.resnet2d(shap_2d, cqt)
    print('3')
    display.resnet2d(guided, cqt)
    print('4')
    display.resnet2d(gc_2d, cqt)


def explanations_aasist(sample):
    aasist_model, device = prepare.load_aasist(weights_AASIST)
    aasist_loader = prepare.dataloader_aasist(sample)
    aasist_pred = predict.aasist(aasist_loader, aasist_model, device)
    state = display.aasist_score(aasist_pred)
    shap_aasist = shap.explain_AASIST(aasist_loader, aasist_model, device, state)
    lime_aasist = LIME.explain_aasist(sample)
    guided_aasist = grad_cam.explain_AASIST_guided(aasist_loader, aasist_model, device, state)
    print('1')
    display.resnet1d(lime_aasist, sample)
    print('2')
    display.resnet1d(shap_aasist, sample)
    print('3')
    display.resnet1d(guided_aasist, sample)


def explain_1d_shap(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    resnet1d_model, device = prepare.load_1d_resnet(weights_1d_resnet)
    resnet1d_loader = prepare.dataloader_1d(sample)
    resnet1d_pred = predict.resnet1d(resnet1d_loader, resnet1d_model, device)
    state = display.resnet1d_score(resnet1d_pred)
    shap_1d = shap.explain_1d_resnet(resnet1d_loader, resnet1d_model, device, state)
    display.resnet1d(shap_1d, sample)
    return()


def explain_1d_gc(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    resnet1d_model, device = prepare.load_1d_resnet(weights_1d_resnet)
    resnet1d_loader = prepare.dataloader_1d(sample)
    resnet1d_pred = predict.resnet1d(resnet1d_loader, resnet1d_model, device)
    state = display.resnet1d_score(resnet1d_pred)
    gc_1d = grad_cam.explain_1d_resnet(resnet1d_loader, resnet1d_model, device, state)
    display.resnet1d(gc_1d, sample)
    return()


def explain_1d_resnet_guided(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    resnet1d_model, device = prepare.load_1d_resnet(weights_1d_resnet)
    resnet1d_loader = prepare.dataloader_1d(sample)
    resnet1d_pred = predict.resnet1d(resnet1d_loader, resnet1d_model, device)
    state = display.resnet1d_score(resnet1d_pred)
    gc_1d = grad_cam.explain_1d_resnet_guided(resnet1d_loader, resnet1d_model, device, state)
    display.resnet1d(gc_1d, sample)
    return()


def explain_1d_lime(database_path, kernel_size=0.5, random_state=41, sample_n=500):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    resnet1d_model, device = prepare.load_1d_resnet(weights_1d_resnet)
    resnet1d_loader = prepare.dataloader_1d(sample)
    resnet1d_pred = predict.resnet1d(resnet1d_loader, resnet1d_model, device)
    _ = display.resnet1d_score(resnet1d_pred)
    lime_1d = LIME.explain_1d_resnet(sample, kernel_size, random_state, sample_n)
    display.resnet1d(lime_1d, sample)
    return lime_1d


def explain_2d_shap(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    cqt = prepare.create_cqt(sample, sr)
    resnet2d_model, device = prepare.load_2d_resnet(weights_2d_resnet)
    resnet2d_loader = prepare.dataloader_2d(cqt)
    resnet2d_pred = predict.resnet2d(resnet2d_loader, resnet2d_model, device)
    state = display.resnet2d_score(resnet2d_pred)
    shap_2d = shap.explain_2d_resnet(resnet2d_loader, resnet2d_model, device, state)
    display.resnet2d(shap_2d, cqt)
    return shap_2d


def explain_2d_gc(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    cqt = prepare.create_cqt(sample, sr)
    resnet2d_model, device = prepare.load_2d_resnet(weights_2d_resnet)
    resnet2d_loader = prepare.dataloader_2d(cqt)
    resnet2d_pred = predict.resnet2d(resnet2d_loader, resnet2d_model, device)
    state = display.resnet2d_score(resnet2d_pred)
    gc_2d = grad_cam.explain_2d_resnet(resnet2d_loader, resnet2d_model, device, state)
    display.resnet2d(gc_2d, cqt)
    return


def explain_2d_guided(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    cqt = prepare.create_cqt(sample, sr)
    resnet2d_model, device = prepare.load_2d_resnet(weights_2d_resnet)
    resnet2d_loader = prepare.dataloader_2d(cqt)
    resnet2d_pred = predict.resnet2d(resnet2d_loader, resnet2d_model, device)
    state = display.resnet2d_score(resnet2d_pred)
    gc_2d = grad_cam.explain_2d_resnet_guided(resnet2d_loader, resnet2d_model, device, state)
    display.resnet2d(gc_2d, cqt)
    return


def explain_2d_lime(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    cqt = prepare.create_cqt(sample, sr)
    lime_2d = LIME.explain_2d_resnet(cqt)
    display.resnet2d_lime(lime_2d, cqt)
    return lime_2d


def explain_aasist_shap(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)

    aasist_model, device = prepare.load_aasist(weights_AASIST)
    aasist_loader = prepare.dataloader_aasist(sample)
    aasist_pred = predict.aasist(aasist_loader, aasist_model, device)
    state = display.aasist_score(aasist_pred)
    shap_aasist = shap.explain_AASIST(aasist_loader, aasist_model, device, state)
    display.AASIST(shap_aasist, sample)
    return


def explain_aasist_lime(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    lime_aasist = LIME.explain_aasist(sample)
    display.AASIST(lime_aasist, sample)
    return


def explain_aasist_guided(database_path):
    sample, sr = sf.read(database_path)
    sample, sr = prepare.tanspose_audio_fragments(sample, sr)
    aasist_model, device = prepare.load_aasist(weights_AASIST)
    aasist_loader = prepare.dataloader_aasist(sample)
    aasist_pred = predict.aasist(aasist_loader, aasist_model, device)
    state = display.aasist_score(aasist_pred)
    gc_aasist = grad_cam.explain_AASIST_guided(aasist_loader, aasist_model, device, state)
    display.AASIST(gc_aasist, sample)
    return

