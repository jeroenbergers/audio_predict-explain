import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import prepare

window = 300

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def predictions(resnet1d, resnet2d, aasist):
    resnet1d = resnet1d.numpy()
    resnet2d = resnet2d.numpy()
    aasist = aasist.numpy()

    #predction from resnet = prediction|bonafide, prediction|fake
    #prediction from aasist = prediction|fake, prediction|bonafide

    #the state determines the target in grad-cam and SHAP. For resnet target = 1 for fake outcome, traget = 0 for
    #bonafide outcome. As the outcome from AASIST is the other way around the target aswell.

    if (resnet1d[0][1] > 0):
        print("Resnet 1d prediction: fake with score: " + str(resnet1d[0][1]))
        state1d = 1
    else:
        print("Resnet 1d prediction: bonafide with score: " + str(resnet1d[0][1]))
        state1d = 0

    if (resnet2d[0][1] > 0):
        print("Resnet 2d prediction: fake with score: " + str(resnet2d[0][1]))
        state2d = 1
    else:
        print("Resnet 2d prediction: bonafide with score: " + str(resnet2d[0][1]))
        state2d = 0

    if (aasist[0][0] > 1.8):
        print("AASIST prediction: fake with score: " + str(aasist[0][0]))
        stateA = 0
    else:
        print("AASIST prediction: bonafide with score: " + str(aasist[0][0]))
        stateA = 1
    return(state1d, state2d , stateA)

def explanations(ex_shap_1d, ex_gc_1d, ex_lime_1d, ex_shap_2d, ex_gc_2d, ex_lime_2d, ex_shap_aasist, ex_lime_aasist, sample, cqt):
    print("1d resnet, SHAP explanation:")
    resnet1d(ex_shap_1d, sample)
    print("1d resnet, grad cam explanation:")
    resnet1d(ex_gc_1d, sample)
    print("1d resnet, lime explanation:")
    resnet1d(ex_lime_1d, sample)

    print("2d resnet, SHAP explanation:")
    resnet2d(ex_shap_2d, cqt)
    print("2d resnet, grad cam explanation:")
    resnet2d(ex_gc_2d, cqt)
    print("2d resnet, lime explanation:")
    resnet2d(ex_lime_2d, cqt)

    print("AASIST, SHAP explanation:")
    AASIST(ex_shap_aasist, sample)
    print("AASIST, lime explanation:")
    AASIST(ex_lime_aasist, sample)


def resnet1d(mask, file):
    a = [0, 16000, 16000 * 2, 16000 * 3, 16000 * 4,16000 * 5,16000 * 6]
    labels = [0, 1, 2, 3, 4, 5, 6]

    limit_mask = np.average(abs(mask)) + 3 * np.std(mask)
    mask[mask < limit_mask] = 0
    normal_gs = moving_average(mask, window)
    print(mask.sum())

    limit = max(np.amax(normal_gs), abs(np.amin(normal_gs)))
    ylimit = max(np.amax(file), abs(np.amin(file)))
    ylimit = ylimit + 0.2 * ylimit
    if (ylimit > 1.1): ylimit = 1.2

    heatmap = np.zeros([1, 96000])
    heatmap = np.expand_dims(normal_gs, axis=0)

    plt.figure(figsize=(18.5, 6.5), dpi=300)
    plt.imshow(np.expand_dims(heatmap, axis=2), cmap='Reds', aspect="auto", interpolation='none',
               extent=[0, 96000, file.min(), file.max()], vmin=0, vmax=limit, alpha=0.5)
    plt.plot(file, label="V_d")
    plt.colorbar()
    plt.grid()

    plt.xlim(-7000, 70000)
    plt.ylim(-ylimit, ylimit)
    plt.xticks(ticks=a, labels=labels)
    plt.ylabel("Hz")
    plt.xlabel("Seconds")
    plt.show()
    return()

def AASIST(mask, file):
    a = [0, 16000, 16000 * 2, 16000 * 3, 16000 * 4]
    labels = [0, 1, 2, 3, 4]

    mask = mask

    limit_mask = np.average(abs(mask)) + 3 * np.std(mask)
    mask[mask < limit_mask] = 0
    normal_gs = moving_average(mask, window)
    print(mask.sum())

    limit = max(np.amax(normal_gs), abs(np.amin(normal_gs)))
    ylimit = max(np.amax(file), abs(np.amin(file)))
    ylimit = ylimit + 0.2 * ylimit
    if (ylimit > 1.1): ylimit = 1.2

    heatmap = np.zeros([1, 64600])
    heatmap = np.expand_dims(normal_gs, axis = 0)

    plt.figure(figsize=(18.5, 6.5), dpi=300)
    plt.imshow(np.expand_dims(heatmap, axis=2), cmap='Reds', aspect="auto", interpolation='none',
               extent=[0, 64600, file.min(), file.max()], vmin=0, vmax=limit, alpha=1)
    plt.plot(file, label="V_d")
    plt.colorbar()
    plt.grid()

    plt.xlim(-7000, 70000)
    plt.ylim(-ylimit, ylimit)
    plt.xticks(ticks=a, labels=labels)
    plt.ylabel("Hz")
    plt.xlabel("Seconds")
    plt.show()
    return()


def resnet2d(mask, sample):
    y = [0, 50, 100, 150, 200, 250, 300]
    x = [0, 66, 2 * 66, 3 * 66, 4 * 66, 5 * 66, 6 * 66]
    yvalues = ['4096', '2048', '1024', '512', '256', '128', '64']
    xvalues = [0, 1, 2, 3, 4, 5, 6]

    file = np.flip(np.flip(sample[120:], 1), 0)
    mask = np.flip(np.flip(mask[120:], 1), 0)
    limit = max(np.amax(mask), abs(np.amin(mask)))

    alpha = np.zeros((312, 400))
    limit_mask = np.average(abs(mask)) + (2 * np.std(mask))
    mask_plus = (mask > limit_mask).astype(int)
    mask_min = (mask < -limit_mask).astype(int)
    alpha = (alpha + mask_plus + 0)

    plt.figure(figsize=(18.5, 6.5), dpi=300)
    plt.imshow(np.expand_dims(file, axis=2), cmap="gist_gray", aspect="auto")
    plt.imshow(np.expand_dims(mask, axis=2), cmap="bwr", aspect="auto", vmin=-limit, vmax=limit, alpha=alpha)

    plt.ylabel("Hz")
    plt.xlabel("Seconds")
    plt.colorbar()
    plt.yticks(y, yvalues)
    plt.xticks(x, xvalues)
    plt.grid()
    plt.show()
    return()