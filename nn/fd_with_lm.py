import numpy as np
import onnxruntime as ort
import cv2
from math import ceil
from itertools import product as product

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        # output = torch.Tensor(anchors).view(-1, 4)
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output.clip(0, 1)
        return output




class Fdlm():

    cfg_mnet = {
        'name': 'mobilenet0.25',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 32,
        'ngpu': 1,
        'epoch': 250,
        'decay1': 190,
        'decay2': 220,
        'image_size': 640,
        'pretrain': True,
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64
    }

    cfg_re50 = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'pretrain': True,
        'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
        'in_channel': 256,
        'out_channel': 256
    }

    def __init__(self,
                 model_path,
                 confidence_threshold = 0.02,
                 top_k = 5000,
                 keep_top_k = 750,
                 nms_threshold = 0.4):

        self.onnx = ort.InferenceSession(model_path)
        self.im_height = 640
        self.im_width = 640
        self.cfg = self.cfg_mnet
        self.priorbox = PriorBox(self.cfg, image_size=(self.im_height, self.im_width))
        self.prior_data = self.priorbox.forward()
        self.scale = np.array([640, 640, 640, 640])
        self.scale1 = np.array([640 for i in range(10)])

        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold

    def run(self, img, rgb_reverse=False):
        # img in BGR
        img = self.preprocess(img, rgb_reverse)
        loc, conf, landms = self.onnx.run(None, {self.onnx.get_inputs()[0].name: img})
        return self.postprocess(loc, conf, landms)

    def preprocess(self, img, rgb_reverse=False):
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_COLOR)

        self.oh, self.ow, _ = img.shape
        if rgb_reverse:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (640, 640))
        # im_height, im_width, _ = img.shape

        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        return img

    def postprocess(self, loc, conf, landms):
        boxes = self.decode(loc.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * self.scale
        scores = conf.squeeze(0)[:, 1]
        landms = self.decode_landm(landms.squeeze(0), self.prior_data, self.cfg['variance'])
        landms = landms * self.scale1

        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        to_o_size = [self.ow / 640, self.oh / 640, self.ow / 640, self.oh / 640]
        bboxes = dets[:, :4] * to_o_size
        lms = landms.reshape((-1, 5, 2)) * to_o_size[:2]

        return bboxes, dets[:, 4], lms

    # Adapted from https://github.com/Hakuyume/chainer-ssd
    def decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self, pre, priors, variances):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                                 ), axis=1)
        return landms

    def draw_img(self, img, boxes, conf, landms, thd = 0.6):
        for b,l,c in zip(boxes, landms, conf):
            if c < thd:
                continue

            text = "{:.4f}".format(c)
            b = list(map(int, b))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            l = l.astype(np.int32)
            cv2.circle(img, (l[0][0], l[0][1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (l[1][0], l[1][1]), 1, (0, 255, 255), 6)
            cv2.circle(img, (l[2][0], l[2][1]), 1, (255, 0, 255), 8)
            cv2.circle(img, (l[3][0], l[3][1]), 1, (0, 255, 0), 10)
            cv2.circle(img, (l[4][0], l[4][1]), 1, (255, 0, 0), 12)

        cv2.imwrite("fd_with_lm.jpg", img)


def CLI(img):

    fd = Fdlm("../models/FaceDetector.onnx")
    img = cv2.imread(img)
    res = fd.run(img)
    fd.draw_img(img, *res)

if __name__ == "__main__":

    import fire
    fire.Fire(CLI)
