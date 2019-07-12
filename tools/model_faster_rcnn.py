from collections import defaultdict
from operator import itemgetter

import numpy as np
from torch.utils.data import DataLoader

from tools.trainer import ModelBase


class ModelRCNN(ModelBase):
    def __init__(self, model, device, metric_iou_treshold=0.75):
        super().__init__(model, device)
        self.metric_iou_treshold = metric_iou_treshold

    def train_batch(self, X, y, optimizer):
        X_dev = list(image.to(self.device) for image in X)
        y_dev = [{k: v.to(self.device) for k, v in t.items()} for t in y]
        loss_dict = self.model(X_dev, y_dev)
        loss_value = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        return {k: v.detach().item() for k, v in loss_dict.items()}

    def eval_batch(self, X, y):
        X_dev = list(image.to(self.device) for image in X)
        y_dev = None if not y else [{k: v.to(self.device) for k, v in t.items()} for t in y]
        result = self.model(X_dev, y_dev)
        result = [{k: v.detach().cpu() for k, v in row.items()} for row in result]
        return result

    def eval_append_results(self, predictions, ground_truth):
        self.ground_truth.extend({k: v.numpy() for k, v in row.items()} for row in ground_truth)
        self.predictions.extend({k: v.numpy() for k, v in row.items()} for row in predictions)

    def evaluate(self, prediction, ground_truth):
        return {'mAP': mAP(ground_truth, prediction, self.metric_iou_treshold)}

    def dataloader_factory(self, dataset, train, batch_size):
        def collate_fn(batch):
            return tuple(list(v) for v in zip(*batch))

        return DataLoader(dataset, batch_size=batch_size, shuffle=train, collate_fn=collate_fn)

    def epoch_metrics_format(self):
        return "LR:{learning_rate:.6f}, loss_cls:{loss_classifier:.4f}, loss_box:{loss_box_reg:.4f}, loss_obj:{loss_objectness:.4f}, loss_rpn_box:{loss_rpn_box_reg:.4f}, mAP:{mAP:.3f}"

    def batch_metrics_format(self):
        return "loss_cls:{loss_classifier:.4f}, loss_box:{loss_box_reg:.4f}, loss_obj:{loss_objectness:.4f}, loss_rpn_box:{loss_rpn_box_reg:.4f}"

    def target_metric(self):
        return "mAP"


def iou(box1, box2):
    x1left, x1right = min(box1[::2]), max(box1[::2])
    y1top, y1bottom = min(box1[1::2]), max(box1[1::2])
    x2left, x2right = min(box2[::2]), max(box2[::2])
    y2top, y2bottom = min(box2[1::2]), max(box2[1::2])
    xi = max(0, min(x1right, x2right) - max(x1left, x2left))
    yi = max(0, min(y1bottom, y2bottom) - max(y1top, y2top))
    intersection = xi * yi
    union = (x1right - x1left) * (y1bottom - y1top) + (x1right - x1left) * (y2bottom - y2top) - intersection
    return intersection / union


def mAP(ground_truth, predictions, iou_threshold):
    all_predictions = defaultdict(list)
    class_counts = defaultdict(int)
    for gt, pred in zip(ground_truth, predictions):
        true_boxes = defaultdict(list)
        for box, label in zip(gt['boxes'], gt['labels']):
            true_boxes[label].append(box)
            class_counts[label] = class_counts[label] + 1
        for box, label, confidence in zip(pred['boxes'], pred['labels'], pred['scores']):
            correct = False
            if label in true_boxes:
                for gt_box in true_boxes[label]:
                    if iou(gt_box, box) >= iou_threshold:
                        correct = True
                        break
            all_predictions[label].append((confidence, correct))
    ap = {k: 0 for k in class_counts}
    for k, v in all_predictions.items():
        v.sort(key=itemgetter(0), reverse=True)
        v = np.array(v)
        recall = np.cumsum(v[:, 1])
        gt_count = class_counts[k]
        cum = 0.0
        if gt_count > 0:
            precision = recall / np.cumsum(np.ones(v.shape[0]))
            recall = recall / gt_count
            r_prev = 0.0
            p_prev = 1.0
            for i in range(recall.shape[0]):
                if precision[i] < p_prev and recall[i] > r_prev:
                    cum += (recall[i] - r_prev) * p_prev
                    r_prev = recall[i]
                p_prev = precision[i]
            cum += (recall[-1] - r_prev) * p_prev
        ap[k] = cum
    return sum(ap.values()) / len(ap) if len(ap) > 0 else 0
