# NMS

```python
import numpy as np


def box_iou(box1, box2):
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    left_top = np.maximum(box1[:,:2],box2[:,:2])
    right_down = np.minimum(box1[:,2:4],box2[:,2:4])
    wh = np.maximum(right_down,left_top,0)
    intersetion_area = wh[:,0]*wh[:,1]
    box1_area = (box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1])
    box2_area = (box2[:,2]-box2[:,0])*(box2[:,3]-box2[:,1])
    iou = intersetion_area/(box2_area+box1_area-intersetion_area+1e-8)
    return iou
    
    
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
            # 获取概率最高的box
            max_ind = np.argmax(cls_boxes[:, 4])
            best_box = cls_boxes[max_ind]
            
            nms_boxes.append(best_box)
            # 排除概率最大的box计算iou
            cls_bboxes = np.concatenate([cls_boxes[:max_ind, :], cls_boxes[max_ind + 1:, :]],axis=0)
            iou = box_iou(best_box, cls_bboxes)
            # 过滤掉重合度较高的box
            iou_mask = iou>thres
            cls_boxes = cls_bboxes[iou_mask]
    return nms_boxes
```