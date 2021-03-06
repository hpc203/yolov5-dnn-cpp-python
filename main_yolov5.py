import cv2
import argparse
import numpy as np

class yolov5():
    def __init__(self, yolo_type, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')    ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)

        self.net = cv2.dnn.readNet(yolo_type + '.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / 640, frameWidth / 640
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold and detection[4] > self.objThreshold:
                    center_x = int(detection[0] * ratiow)
                    center_y = int(detection[1] * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame
    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = outs[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # outs[i] = outs[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            outs[i] = outs[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            if self.grid[i].shape[2:4] != outs[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-outs[i]))  ### sigmoid
            ###其实只需要对x,y,w,h做sigmoid变换的， 不过全做sigmoid变换对结果影响不大，因为sigmoid是单调递增函数，那么就不影响类别置信度的排序关系，因此不影响后面的NMS
            ###不过设断点查看类别置信度，都是负数，看来有必要做sigmoid变换把概率值强行拉回到0到1的区间内
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, self.no))
        z = np.concatenate(z, axis=1)
        return z

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='bus.jpg', help="image path")
    parser.add_argument('--net_type', default='yolov5s', choices=['yolov5s', 'yolov5l', 'yolov5m', 'yolov5x'])
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--objThreshold', default=0.5, type=float, help='object confidence')
    args = parser.parse_args()

    yolonet = yolov5(args.net_type, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold, objThreshold=args.objThreshold)
    srcimg = cv2.imread(args.imgpath)
    dets = yolonet.detect(srcimg)
    srcimg = yolonet.postprocess(srcimg, dets)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
