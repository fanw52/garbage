#NMS


```python
def nms(boxes):
    '''
    [x1,y1,x2,y2,score,cls]
    '''
    # 获取每个类别的box
    output_box = []
    for i in range(n_class):
        cur_box = box[...,cls==i]
            # 按照score对box排序
        cur_box = sorted(cur_box,lambda item:[item[4]],reverse=True)
            nms_box = []
            # 遍历所有的box,按照一定的iou过滤
        # for _box in cur_box[1:]:
        while cur_box:
          ibox = cur_box[0]
          x = iou(ibox,cur_box)
          cur_box = cur_box[x>thres]
          nms_box.append(ibox)
       output_box.append(nms_box)
   return output_box
```