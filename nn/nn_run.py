#!/usr/bin/env python
# coding: utf-8

import onnxruntime as ort
import cv2
import numpy as np


def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    #print(boxes)
    #print(confidences)

    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        #print(confidences.shape[1])
        probs = confidences[:, class_index]
        #print(probs)
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        #print(subset_boxes)
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


class NNRun():
    pass

class Fd(NNRun):

    def __init__(self, model_path: str):
        self.onnx = ort.InferenceSession(model_path)
        self.image_mean = np.array([127, 127, 127])
        self.image_scale = 128
        self.input_size = (self.onnx.get_inputs()[0].shape[3], self.onnx.get_inputs()[0].shape[2])

    # scale current rectangle to box
    def scale(self, box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        maximum = max(width, height)
        dx = int((maximum - width) / 2)
        dy = int((maximum - height) / 2)

        x1 = np.clip(box[0] - dx, 0, self.o_shape[1])
        y1 = np.clip(box[1] - dy, 0, self.o_shape[0])
        x2 = np.clip(box[2] + dx, 0, self.o_shape[1])
        y2 = np.clip(box[3] + dy, 0, self.o_shape[0])

        bboxes = [x1, y1, x2, y2]
        return bboxes

    def run(self, orig_image, rgb_reverse=1, threshold=0.3):

        if isinstance(orig_image, str):
            orig_image = cv2.imread(orig_image)

        if rgb_reverse:
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        else:
            image = orig_image

        self.o_shape = orig_image.shape

        image = cv2.resize(image, self.input_size)
        image = (image - self.image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.onnx.get_inputs()[0].name
        confidences, boxes = self.onnx.run(None, {input_name: image})
        boxes, labels, probs = predict(self.o_shape[1], self.o_shape[0], confidences, boxes, threshold)
        rects = []
        for box in boxes:
            rects.append(self.scale(box))
        return rects, labels, probs

class Fr(NNRun):

    def __init__(self, model_path: str):
        self.onnx = ort.InferenceSession(model_path)
        self.image_mean = np.array([127, 127, 127])
        self.image_scale = 128
        self.input_size = (self.onnx.get_inputs()[0].shape[3], self.onnx.get_inputs()[0].shape[2])

    def run(self, orig_image, rgb_reverse=1):
        # image RGB
        if isinstance(orig_image, str):
            orig_image = cv2.imread(orig_image)

        if rgb_reverse:
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        else:
            image = orig_image

        image = cv2.resize(image, self.input_size)
        image = (image - self.image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        input_name = self.onnx.get_inputs()[0].name
        emb = self.onnx.run(None, {input_name: image})
        emb = emb[0].squeeze()

        s = np.sqrt(sum(pow(emb, 2)))
        res = emb / s

        return res

class CLI():
    color = (255, 128, 0)
    def fd(self, image):

        nn_fd = Fd("../models/version-RFB-320-opt.onnx")
        img = cv2.imread(image)
        boxes, _, _ = nn_fd.run(img)

        for box in boxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), self.color, 4)
        cv2.imwrite('./fd_out.jpg', img)

    def fdfr(self, image):
        nn_fd = Fd("../models/version-RFB-320-opt.onnx")
        nn_fr = Fr("../models/mobilefacenet.onnx")
        img = cv2.imread(image)
        boxes, _, _ = nn_fd.run(img)

        for box in boxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), self.color, 4)
            print(box)
            img_crop = img[box[1]: box[3], box[0]: box[2]]
            emb = nn_fr.run(img_crop)
            print(emb)
            cv2.imwrite("%d.jpg"%box[1], img_crop)

        cv2.imwrite('./fdfr_out.jpg', img)

if __name__ == "__main__":
    import fire
    fire.Fire(CLI())