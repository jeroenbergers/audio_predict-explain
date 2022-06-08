import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import prepare

window = 100


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def predictions(resnet1d_score, resnet2d_score, aasist_score):
    # prediction from resnet = prediction|bonafide, prediction|fake
    # prediction from aasist = prediction|fake, prediction|bonafide

    # the state determines the target in grad-cam and SHAP. For resnet target = 1 for fake outcome, traget = 0 for
    # bonafide outcome. As the outcome from AASIST is the other way around the target aswell.

    if resnet1d_score[0][1] > 0:
        print("Resnet 1d prediction: fake with score: " + str(resnet1d_score[0][1]))
        state1d = 1
    else:
        print("Resnet 1d prediction: bonafide with score: " + str(resnet1d_score[0][1]))
        state1d = 0

    if resnet2d_score[0][1] > 0:
        print("Resnet 2d prediction: fake with score: " + str(resnet2d_score[0][1]))
        state2d = 1
    else:
        print("Resnet 2d prediction: bonafide with score: " + str(resnet2d_score[0][1]))
        state2d = 0

    if aasist_score[0][0] > 1.8:
        print("AASIST prediction: fake with score: " + str(aasist_score[0][0]))
        state_a = 0
    else:
        print("AASIST prediction: bonafide with score: " + str(aasist_score[0][0]))
        state_a = 1
    return state1d, state2d, state_a


def resnet1d_score(score):
    if score[0][1] > 0:
        print("Resnet 1d prediction: fake with score: " + str(score[0][1]))
        state = 1
    else:
        print("Resnet 1d prediction: bonafide with score: " + str(score[0][1]))
        state = 0

    return state


def aasist_score(score):
    if score[0][0] > 1.8:
        print("AASIST prediction: fake with score: " + str(score[0][0]))
        state = 0
    else:
        print("AASIST prediction: bonafide with score: " + str(score[0][0]))
        state = 1
    return state


def resnet2d_score(score):
    if score[0][1] > 0:
        print("Resnet 2d prediction: fake with score: " + str(score[0][1]))
        state = 1
    else:
        print("Resnet 2d prediction: bonafide with score: " + str(score[0][1]))
        state = 0
    return state


def explanations(ex_shap_1d, ex_gc_1d, ex_lime_1d, ex_shap_2d, ex_gc_2d, ex_lime_2d, ex_shap_aasist, ex_lime_aasist,
                 sample, cqt):
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
    # a = [0, 16000, 16000 * 2, 16000 * 3, 16000 * 4,16000 * 5,16000 * 6]
    # labels = [0, 1, 2, 3, 4, 5, 6]

    limit_mask = np.average(abs(mask)) + 1 * np.std(mask)
    mask[mask < limit_mask] = 0
    normal_gs = moving_average(mask, window)

    xvalues, x = labels_1d(file, 16000)
    limit = max(np.amax(normal_gs), abs(np.amin(normal_gs)))
    ylimit = max(np.amax(file), abs(np.amin(file)))
    ylimit = ylimit + 0.2 * ylimit
    if ylimit > 1.1:
        ylimit = 1.2

    heatmap = np.expand_dims(normal_gs, axis=0)

    plt.figure(figsize=(12, 5), dpi=300)
    plt.imshow(np.expand_dims(heatmap, axis=2), cmap='Reds', aspect="auto", interpolation='none',
               extent=[0, mask.shape[0], file.min(), file.max()], vmin=0, vmax=limit, alpha=0.5)
    plt.plot(file, label="V_d")
    plt.colorbar()
    plt.grid()

    plt.xlim(-7000, mask.shape[0])
    plt.ylim(-ylimit, ylimit)
    plt.xticks(ticks=xvalues, labels=x)
    plt.ylabel("dB")
    plt.xlabel("Seconds")
    plt.show()
    return ()


def AASIST(mask, file):

    mask = mask
    labels, x = labels_1d(file, 16000)

    limit_mask = np.average(abs(mask)) + 3 * np.std(mask)
    mask[mask < limit_mask] = 0
    normal_gs = moving_average(mask, window)
    print(mask.sum())
    limit = max(np.amax(normal_gs), abs(np.amin(normal_gs)))
    ylimit = max(np.amax(file), abs(np.amin(file)))
    ylimit = ylimit + 0.2 * ylimit
    if ylimit > 1.1:
        ylimit = 1.2

    heatmap = np.expand_dims(normal_gs, axis=0)

    plt.figure(figsize=(12, 5), dpi=300)
    plt.imshow(np.expand_dims(heatmap, axis=2), cmap='Reds', aspect="auto", interpolation='none',
               extent=[0, mask.shape[0], file.min(), file.max()], vmin=0, vmax=limit, alpha=1)
    plt.plot(file, label="V_d")
    plt.colorbar()
    plt.grid()

    plt.xlim(-7000, mask.shape[0])
    plt.ylim(-ylimit, ylimit)
    plt.xticks(labels, x)
    plt.ylabel("dB")
    plt.xlabel("Seconds")
    plt.savefig("AASIST.png", dpi=500)
    plt.show()
    return ()


def resnet2d(input, sample):
    y = [0, 50, 100, 150, 200, 250, 300]
    yvalues = ['8000', '4000', '2000', '1000', '500', '250', '125']

    xvalues, x = labels_2d(sample, 66)

    file = np.fliplr(np.fliplr(np.flipud(sample[120:])))
    mask = np.fliplr(np.fliplr(np.flipud(input[120:])))
    limit = max(np.amax(mask), abs(np.amin(mask)))

    alpha = np.zeros(mask.shape)
    limit_mask = np.average(abs(mask)) + (2 * np.std(mask))
    mask_plus = (mask > limit_mask).astype(int)
    alpha = (alpha + mask_plus + 0)

    plt.figure(figsize=(12, 5), dpi=300)
    plt.imshow(np.expand_dims(file, axis=2), cmap="gist_gray", aspect="auto")
    plt.imshow(np.expand_dims(mask, axis=2), cmap="bwr", aspect="auto", vmin=-limit, vmax=limit, alpha=alpha)

    plt.ylabel("Hz")
    plt.xlabel("Seconds")
    plt.colorbar()
    plt.yticks(y, yvalues)
    plt.xticks(xvalues, x)
    plt.grid()
    plt.savefig("resnet2d.png", dpi=500)
    plt.show()
    return ()


def labels_1d(sample, fs):
    lent = sample.shape[0]
    number_ticks = int(lent / fs)
    x = np.arange(0, number_ticks + 1, 1, dtype=int)
    x_values = np.arange(0, fs * (number_ticks + 1), fs, dtype=int)
    return x_values, x


def labels_2d(sample, fs):
    lent = sample.shape[1]
    number_ticks = int(lent / fs)
    x = np.arange(0, number_ticks + 1, 1, dtype=int)
    x_values = np.arange(0, fs * (number_ticks + 1), fs, dtype=int)
    return x_values, x


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def bulk_1d(mask_list):
    important_time = normalized(mask_list).sum(axis=0)
    important_time[important_time < 0] = 0
    move = moving_average(important_time, 300)
    heatmap = move
    plt.figure(figsize=(7, 5), dpi=300)
    a = [0, 16000, 16000 * 2, 16000 * 3, 16000 * 4, 16000 * 5, 16000 * 6]
    labels = [0, 1, 2, 3, 4, 5, 6]
    limit = max(np.amax(heatmap), abs(np.amin(heatmap)))
    plt.xticks(ticks=a, labels=labels)
    plt.imshow(np.expand_dims(heatmap, axis=1).T, cmap='Reds', aspect="auto", vmin=0, vmax=limit)
    plt.colorbar()
    plt.savefig("bulk1d.png", dpi=500)
    plt.show()


def bulk_2d(mask_list):
    mask_list = abs(mask_list)
    heatmap = mask_list.sum(axis=0)
    # heatmap = np.rot90(important_time[0][120:], -1)
    heatmap = np.rot90(heatmap[120:], -1)

    plt.figure(figsize=(7, 5), dpi=300)

    y = [0, 50, 100, 150, 200, 250, 300]
    x = [0, 66, 2 * 66, 3 * 66, 4 * 66, 5 * 66, 6 * 66]
    yvalues = ['8000', '4000', '2000', '1000', '500', '250', '125']

    xvalues = [0, 1, 2, 3, 4, 5, 6]

    limit = max(np.amax(heatmap), abs(np.amin(heatmap)))

    plt.imshow(heatmap.T, cmap='Reds', aspect="auto", vmin=0, vmax=limit)
    plt.colorbar()
    plt.yticks(y, yvalues)
    plt.xticks(x, xvalues)
    plt.ylabel("Hz")
    plt.xlabel("Seconds")
    plt.grid()
    plt.savefig("bulk2d.png", dpi=500)
    plt.show()


def resnet2d_lime(input, sample):
    y = [0, 50, 100, 150, 200, 250, 300]
    yvalues = ['8000', '4000', '2000', '1000', '500', '250', '125']

    xvalues, x = labels_2d(sample, 66)

    file = np.fliplr(np.fliplr(np.flipud(sample[120:])))
    mask = np.fliplr(np.fliplr(np.flipud(input[120:])))
    limit = max(np.amax(mask), abs(np.amin(mask)))

    alpha = np.zeros(mask.shape)
    limit_mask = np.average((mask)) + (2 * np.std(mask))
    mask_plus = (mask > limit_mask).astype(int)
    alpha = (alpha + mask_plus + 0)
    print(limit_mask)
    print(mask_plus)

    plt.figure(figsize=(12, 5), dpi=300)
    plt.imshow(np.expand_dims(file, axis=2), cmap="gist_gray", aspect="auto")
    plt.imshow(np.expand_dims(mask, axis=2), cmap="Reds", aspect="auto", vmin=-limit * 0.3, vmax=0, alpha=alpha)

    plt.ylabel("Hz")
    plt.xlabel("Seconds")
    plt.colorbar()
    plt.yticks(y, yvalues)
    plt.xticks(xvalues, x)
    plt.grid()
    plt.show()
    return ()
