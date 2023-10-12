import torch
import cv2
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import subprocess
from hyperparams import hypers

def show_bbox(img,boxes,axis,color=(0,255,0)):
    img=img.copy()
    for i,box in enumerate(boxes):
        box = [ round(elem) for elem in box ]
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color,2)
        y=box[1]-10 if box[1]-10>10 else box[1]+10

    axis.imshow(img)
    axis.axis("off")
    
def extract_box(img_dict):
  boxes=[]
  for key in img_dict['image']:
    boxes.append([img_dict['xmin'][key],img_dict['ymin'][key],img_dict['xmax'][key],img_dict['ymax'][key]])
  return boxes

def create_annotations(test_imgs_dir):
  test_img_list = os.listdir(test_imgs_dir)
  test_label_list = [s.replace(".jpg", ".txt") for s in test_img_list]

  return pd.DataFrame(data = {'image': test_img_list, 'text':test_label_list})

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, S, S, 4) #4 because only 1 box is passed per time
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, S, S, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # print("boxes_preds: ", boxes_preds.shape)
    # print("boxes_labels: ", boxes_labels.shape)

    if box_format == "midpoint" :
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2  # x - (w / 2)
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2  # y - (h / 2)
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2  # x + (w / 2)
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2  # y + (h / 2)

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners" :
        box1_x1, box1_y1 = boxes_preds[..., 0:1], boxes_preds[..., 1:2]
        box1_x2, box1_y2 = boxes_preds[..., 2:3], boxes_preds[..., 3:4]
        box2_x1, box2_y1 = boxes_labels[..., 0:1], boxes_labels[..., 1:2]
        box2_x2, box2_y2 = boxes_labels[..., 2:3], boxes_labels[..., 3:4]

    x1, y1 = torch.max(box1_x1, box2_x1), torch.max(box1_y1, box2_y1) #greater upper-left corner
    x2, y2 = torch.min(box1_x2, box2_x2), torch.min(box1_y2, box2_y2) #lower bottom-right corner


    '''
    limits the range to a lower bound of 0, so to avoid negative values.
    inter computes the area, how?
    '''

    inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    box1_area = abs( (box1_x2 - box1_x1) * (box1_y2 - box1_y1) ) #base * altezza
    box2_area = abs( (box2_x2 - box2_x1) * (box2_y2 - box2_y1) )

    uni = box1_area + box2_area - inter

    return inter / (uni + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    avg_precisions = []
    epsilon = 1e-6 # used for numerical stability
    for c in range(num_classes):
        detections, ground_truths = [], []
        for detection in pred_boxes:
            if detection[1] == c: detections.append(detection)
        for true_box in true_boxes :
            if true_box[1] == c: ground_truths.append(true_box)
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key,val in amount_bboxes.items(): amount_bboxes[key] = torch.zeros(val)
        detections.sort(key=lambda x:x[2], reverse=True)
        TP, FP, total_true_bboxes = torch.zeros(len(detections)), torch.zeros(len(detections)), len(ground_truths)
        if total_true_bboxes == 0:
            continue
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [ bbox for bbox in ground_truths if bbox[0] == detection[0] ]
            num_gts = len(ground_truth_img)
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img) :
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format,)
                if iou > best_iou :
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_threshold :
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else :
                    FP[detection_idx] = 1
            else :
                FP[detection_idx] = 1
        TP_cumsum, FP_cumsum = torch.cumsum(TP, dim=0), torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions, recalls = torch.cat((torch.tensor([1]), precisions)), torch.cat((torch.tensor([0]), recalls))
        avg_precisions.append(torch.trapz(precisions, recalls))

    return sum(avg_precisions) / (len(avg_precisions) + epsilon)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width, box[3] * height,
            linewidth=1, edgecolor="r", facecolor="none",
        )
        ax.add_patch(rect)
    plt.show()

def get_bboxes(loader, model, iou_threshold, threshold,S=7, B=2, C=1, pred_format="cells", box_format="midpoint", device="cuda"):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels,S,B,C)
        bboxes = cellboxes_to_boxes(predictions,S,B,C)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7,B=2, C=20):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """
    num_classes=C
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, B*5+C)
    bboxes1 = predictions[..., num_classes+1:num_classes+5]
    bboxes2 = predictions[..., num_classes+6:num_classes+10]
    scores = torch.cat(
        (predictions[..., num_classes].unsqueeze(0), predictions[..., num_classes+5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :num_classes].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., num_classes], predictions[..., num_classes+5]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7,B=2, C=20):
    converted_pred = convert_cellboxes(out,S,B,C).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []
    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def makeCSVfile(dataset, folder):
    images = subprocess.check_output("ls "+ dataset.location + folder + '/images', shell=True, text=True).split('\n')[:-1]
    labels = subprocess.check_output("ls "+ dataset.location + folder + '/labels', shell=True, text=True).split('\n')[:-1]
    return pd.DataFrame(data = {'image': images, 'text':labels})

def save_checkpoint(state, filename="my_checkpoint.pth.tar", exit_training=False):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    if exit_training : exit()

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])