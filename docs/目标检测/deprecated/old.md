
目标检测任务个人把它分为如下几块：backbone,neck,postprocession,loss,others
* backbone
    * 常见的结构包括VGG,ResNet,DarkNet,DenseNet等，如果应用在移动端或者CPU上常用的结构有SqueezeNet,ShuffleNet,MobileNet.
    * darknet19:相比于vgg19中间层引入了1x1的卷积层对特征进行压缩，降低网络的计算量，减少网络的参数，因此yolov2会比较快。
    * darknet53:引入残差结构，相比于resnet，该网络结构会利用1x1的卷积压缩特征，压缩后的特征再进行后续的1x1，3x3的卷积操作，降低网络的计算量，减少网络参数。
    * cspdarknet53：cross stage partial network，通过将输入特征划分成两个部分（可视作输入特征在低维空间的两个复制），进行不同的卷积操作，最后合并来减少计算量（作者认为优化过程中梯度信息重复），yolov4的backbone。     
    * senet: squeeze and excitation,global average pooling 做squeeze操作，exciatation通过两层全连接层体现，并用sigmoid体现每个channel特征的重要性,yolov4中的SAM是类似的思想。
    * mobilenetv2: inverted residuals and linear bottlenecks，inverted residuals:与普通的瓶颈结构相反，这里先使用1x1的卷积对输入特征进行扩张，避免在depthwise conv过程中输出过多无用的信息（实验发现，depthwise conv训练出来的kernel有不少是空的，那么其对应的特征也就成为了一个无效的特征，出现特征退化的现象，外加relu，特征退化的就回更明显了，因此先扩展缓解特征退化的现象，），linear bottlenecks，将elements+ 操作后的relu去掉，减少relu对特征的破坏程度。
    * SqueezeNet: group conv + channel shuffle,group conv是depthwise conv的一种泛化版本，该方法将若干个通道划分成一个独立的组，然后组内分别计算，但是这样会让组间的特征无法交互，因此，通过引入通道混洗的方法一定程度上弥补这个问题，通道混洗实际就是对输入数据做reshape，然后做转置，最后平摊分组、比如：我们将输入数据分成g组，每组有n个通道，第一步需要将输入数据reshape成 （g,n）的形式，这里不考虑空间维度，然后，我们将reshape后的数据transpose为（n,g）的形式，表示n组，每组g个通道，这样就会保证转置后的每个分组会包含分组前的组内的一个特征，随后我们将(n,g)的矩阵平摊回去，就完成了不同组之间的信息交互。
    * SqueezeNetv2:该论文探索了影响模型实际执行效率的原因，比较重要的一点就是MAC(内存访问代价)，当输入通道与输出通道相等时，内存访问代价最小，elemwise的操作会增加MAC,因此使用concat 替代elem-wise+操作，因此输入数据在进行block之前，会实现对输入数据进行平分，然后无论是使用group conv还是depth-wise conv都会降低MAC。
* neck
    * FPN：feature pyramid network,top-down机制，高层特征通过上采样，扩充尺寸与浅层特征融合，进行目标检测与目标回归任务。
    * PAN:top-down+bottom-up机制。在top-down机制的基础上，将浅层特征下采样，与深层特征再次的进行融合。
    * SPP ,spatial pyramid pooling，解决了因图片尺寸不同而无法进行优化的问题，通过引入几组不同的kernel，将目标特征池化到固定的尺寸，然后连接fc层做后续的任务，这里通过不同的kernel获取不同尺寸的输出，并将各个输出结合起来，作为最终的表示，这种方式从多个角度去描述输入特征，会提高网络在物体发生形变或缩放时的鲁棒性。
    * RoI pooling:类似于SPP，RoI pooling的的目的是将一种图片上的某个目标的特征固定到指定尺寸，解决了一张图片上，多个不同尺寸的目标的优化问题。
    * ROI Align：目标检测中，用于回归与分类的目标特征是通过原始图片与特征图片位置关系的映射得到的，而在映射过程中，目标在特征图上的映射得到的位置不一定处在特征图的整点位置，当采用取整操作获取目标特征时，会带来一定的信息损失，而RoI Align采用双线性插值的方法，保留了这些无法落在整点上的特征信息。
    * SAM：spatial attention module，是一种特征级别的注意力机制，会给每个channel计算出一个注意力权重，实现方式:通过global average pooling获取1x1xC的特征向量，并连接两层全连接层，第一层特征降维，第二层特征升维，连接sigmoid获取每个channel的权重，降维与升维的过程是各个通道特征信息交互的过程，这一点在SENet中得到体现，在Yolov4中，将这种逐通道的注意力机制改变为逐点的注意力机制。
    * anchor的生成：主要包含两个主要的方法，一种是YoLo中提出的以k-means的方法，以数据集的聚类中心作为预定的anchor。另外一种是SSD中的方法，通过固定的面积值，以指定的不同的长宽（SSD中指定了几组(min,max)）以及ratio（0.5,1,2）生成若干个anchor，faster rcnn中类似。
    * ssd与yolo对目标的表示:ssd中回归的是目标中心的偏移量以及长宽的缩放量(cx = x+bx)，而yolo中通过sigmiod限制了偏移量的上下限(cx = sigmod(x)+bx)，可以防止预测异常导致优化不稳定。做出限制就表明目标框中心点的偏移被限制在一定范围内了，但是这并不影响，因为总有一个正样本中心的偏移量会在这个范围内。
* postprocession
    * nms与soft-nms的区别？
        *非极大值抑制，第一种又称hard-nms，二者的不同之处在于后者通过引入与iou相关的高斯函数作为权重系数降低box的score：
        weight = np.exp(-(1.0 * iou ** 2 / sigma)) cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight 
        soft-nms相比nms的优势在于可以缓解部分遮挡的物体，例如行人检测场景，人与人之间存在部分遮挡，但是对于某些特定的场景，nms会更有效，比如文字检测，因为文字基本不存在部分重叠的情况。
* loss
    * smooth l1 loss:常用于目标检测的回归损失，将目标位置与回归位置的偏移量作为参数，进行优化，缺点是我们在训练过程中是以点的偏移量作为损失进行网络优化的，而在推理阶段采用IoU作为衡量标准，选取合适的输出目标，这里存在训练与测试不一致的情况。
    * IoU loss:  以目标框和检测框的重叠面积的作为度量loss的指标，重叠面积越大，损失越小，但问题是，当目标框与检测框没有重叠时，无论相隔多远，loss恒定, 梯度为0，丧失了优化的方向。
    * GIoU loss: GIoU loss引入了 两个box的最小外包围矩形的概念，通过计算两个box的并集在最小外包围矩形中的差集与最小外包围的占比反映loss，差集占比越小，loss越小。通过差集与最小外包围面积，就避免了两个box不重叠时，梯度为0的情况。
    * DIoU loss:  通过两个box的中心距离，解决box不重叠的情况。此外用中心距离的好处在于当一个box包含与另外一个box中，最小GIoU loss便退化为IoU loss了，此时被包含的box的相对位置无法反应在损失函数中，而中心距离就可以体现相对位置。
    * CIoU loss:  目标拟合的效果，不仅要看两个box的中心点，还要看两个box的长宽比，CIoU loss在中心距离的基础上增加了长宽比这一项，长宽比越类似，检测框相对会更好。
others
    * CmBN：cross mini batch normalization,在实际的使用过程中，会收集若干个batch的参数再更新，比如累积4个mini-batch的权重，scale ,shift 一起更新，变相的增加了batch size。类似的思想还有Mosacia augmentation，将四张图片concat在一起，变相地增加输入的batch size。
    * mAP: mean avearge pooling，目标检测performance度量的方式，是多个AP的平均值，而AP是P-R曲线所包围的面积，P-R曲线绘制方式：1、对每个目标根据置信度进行排序，计算大于等于该置信度目标的precision和recall，遍历每一个样本，绘制p-r曲线。
    * STN：spatial transformer network，对倾斜目标进行alignment操作，校正倾斜目标，opencv中有类似的方法，仿射变换。