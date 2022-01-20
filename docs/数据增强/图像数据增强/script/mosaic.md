```python
## copy from yolov5

import math
import random

import cv2
import numpy as np
import torch

class Mosaic():
    def __init__(self, size, p,degrees, translate, scale, shear, perspective):
        self.width, self.height = size
        self.mosaic_border = [-self.width // 2, -self.height // 2]
        self.p = p

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def xywhn2xyxy(self, x, w=640, h=640, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
        return y

    def xyn2xy(self, x, w=640, h=640, padw=0, padh=0):
        # Convert normalized segments into pixel segments, shape (n,2)
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = w * x[:, 0] + padw  # top left x
        y[:, 1] = h * x[:, 1] + padh  # top left y
        return y

    def __call__(self, data: dict):
        # loads images in a 4-mosaic
        if self.p < random.random():
            data.pop('mosaic_data')
            return data
        mosaic_data = data['mosaic_data']
        mosaic_data.append(data)
        labels4, segments4 = [], []
        s = [self.width, self.height]
        yc, xc = [int(random.uniform(-x, 2 * s[i] + x)) for i, x in enumerate(self.mosaic_border)]  # mosaic center x, y
        # indices = [index] + random.choices(range(0,data_nums), k=3)  # 3 additional image indices
        for i, line in enumerate(mosaic_data):
            # Load image
            img, _, (h, w), r = self.load_image(line['img_path'])
            text_polys = line['text_polys'] * r
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s[1] * 2, s[0] * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[0] * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[1] * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[0] * 2), min(s[1] * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            # labels, segments = self.labels[index].copy(), self.segments[index].copy()
            segments = []
            if text_polys.size:
                text_polys[:, :, 0] += padw
                text_polys[:, :, 1] += padh
                # labels[:, 1:] = self.xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [self.xyn2xy(x, w, h, padw, padh) for x in segments]

            labels4.append(text_polys)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4, *segments4):
            np.clip(x[:, :, 0], 0, 2 * s[0], out=x[:, :, 0])  # clip when using random_perspective()
            np.clip(x[:, :, 1], 0, 2 * s[1], out=x[:, :, 1])  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate
        # Augment
        img4, labels4 = random_perspective(img4, labels4, segments4,
                                           degrees=self.degrees,
                                           translate=self.translate,
                                           scale=self.scale,
                                           shear=self.shear,
                                           perspective=self.perspective,
                                           border=self.mosaic_border)  # border to remove

        data['img'] = img4
        data['text_polys'] = labels4.reshape([-1,4,2])
        # TODO:
        n =  labels4.shape[0]

        data['ignore_tags'] = [False for _ in range(n)]
        data['texts'] = ['who_care' for _ in range(n)]
        data.pop('mosaic_data')
        return data

    def load_image(self, path):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.width / max(h0, w0)  # resize image to img_size
        # r = min(self.width / w0, self.height / h0)
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2], r  # img, hw_original, hw_resized


def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets.reshape(n * 4, 2)
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
            # clip
            new = xy
            new[:, [0, 2, 4, 6]] = new[:, [0, 2, 4, 6]].clip(0, width)
            new[:, [1, 3, 5, 7]] = new[:, [1, 3, 5, 7]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets.reshape([-1, 8]).T, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        # targets = targets[i]
        targets = new[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    # w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w1 = (box1[2] - box1[0] + box1[4] - box1[6]) / 2
    h1 = (box1[7] - box1[1] + box1[5] - box1[3]) / 2

    # w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    w2 = (box2[2] - box2[0] + box2[4] - box2[6]) / 2
    h2 = (box2[7] - box2[1] + box2[5] - box2[3]) / 2
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy

```