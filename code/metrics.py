import numpy as np

SMOOTH = 1e-6
from sklearn.metrics import jaccard_score


def metrics(output, target):

    dice_result = []
    iou_result = []

    for i in range(output.shape[0]):

        k = 1
        print(np.sum(output[i][target[i] == k]) )
        print(np.sum(output[i]) )
        print( np.sum(target[i]))

        dice = round(np.sum(output[i][target[i] == k]) * 2.0 / (np.sum(output[i]) + np.sum(target[i])),3)
        dice_result += [dice]


        output_i = output[i].reshape(output.shape[1] * output.shape[2])
        target_i = target[i].reshape(target.shape[1] * target.shape[2])
        iou = round(jaccard_score(target_i, output_i, labels=np.unique(target_i), pos_label=1, average='binary'), 3)



        iou_result += [iou]


    return dice_result, iou_result

