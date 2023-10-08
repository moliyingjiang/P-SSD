import logging
import os
from datetime import datetime

import numpy as np

from .eval_detection_voc import eval_detection_voc


def voc_evaluation(dataset, predictions, output_dir, iteration=None):
    class_names = dataset.class_names

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_labels_list = []
    gt_difficults = []

    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, gt_labels, is_difficult = annotation
        gt_boxes_list.append(gt_boxes)
        gt_labels_list.append(gt_labels)
        gt_difficults.append(is_difficult.astype(np.bool))

        img_info = dataset.get_img_info(i)
        prediction = predictions[i]
        prediction = prediction.resize((img_info['width'], img_info['height'])).numpy()
        boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']

        pred_boxes_list.append(boxes)
        pred_labels_list.append(labels)
        pred_scores_list.append(scores)
    result = eval_detection_voc(pred_bboxes=pred_boxes_list,
                                pred_labels=pred_labels_list,
                                pred_scores=pred_scores_list,
                                gt_bboxes=gt_boxes_list,
                                gt_labels=gt_labels_list,
                                gt_difficults=gt_difficults,
                                iou_thresh=0.5,
                                use_07_metric=True)
    logger = logging.getLogger("SSD.inference")
    result_str = "mAP: {:.4f}\n".format(result["map"])
    metrics = {'mAP': result["map"]}
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        metrics[class_names[i]] = ap
        result_str += "{:<16} \nap is: {:.4f}\n\n".format(class_names[i], ap)
        # result_str += "{} \nap is: {:.4f}\nprecision is {:.4f}\nrecall is P{:.4f}\n\n".format(class_names[i], ap, result["precision"][i], result["recall"][i])
    # print(result_p_r)
    i = len(result["recall"]) - 1
    end_prec = []
    end_rec = []
    while i:
        # pass
        end_prec.append(np.nanmean(result["precision"][i]))
        end_rec.append(np.nanmean(result["recall"][i]))
        print(("class: {:<8}\n  prec is {:.4}\n  recall is {:.4}\n\n").format(class_names[i],np.nanmean(result["precision"][i]),np.nanmean(result["recall"][i])))
        i -= 1
    print(("All is :\n  all_prec is {:.4}\n  all_rec is {:.4}\n\n").format(np.nanmean(end_prec),np.nanmean(end_rec)))
    logger.info(result_str)

    if iteration is not None:
        result_path = os.path.join(output_dir, 'result_{:07d}.txt'.format(iteration))
    else:
        result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write(result_str)

    return dict(metrics=metrics)
