# NMS

```python
import numpy as np


def box_iou(box1, box2):
    return


def nms(boxes,thres):
    '''
    [x1,y1,x2,y2,score,cls]
    '''
    # 获取每个类别的box
    nms_boxes = []
    classes_in_img = list(set(boxes[:, 5]))
    for cls in classes_in_img:
        cls_mask = (boxes[:, 5] == cls)
        cls_boxes = boxes[cls_mask]
        while len(cls_boxes):
            max_ind = np.argmax(cls_boxes[:, 4])
            best_box = cls_boxes[max_ind]
            nms_boxes.append(best_box)
            cls_bboxes = np.concatenate([cls_boxes[:max_ind, :], cls_boxes[max_ind + 1:, :]],axis=0)
            iou = box_iou(best_box, cls_bboxes)
            iou_mask = iou>thres
            cls_boxes = cls_bboxes[iou_mask]
    return nms_boxes
```