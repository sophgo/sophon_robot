import argparse
import cv2
import numpy as np
import sophon.sail as sail
import os
import time

from utils.colors import _COLORS
from utils.utils import *

import threading
import queue

opt = None
save_path = os.path.join(os.path.dirname(
    __file__), "output", os.path.basename(__file__).split('.')[0])

QUEUE_SIZE = 1

class YOLOV5_BMCV_Detector2(object):
    def __init__(self, bmodel_path, tpu_id, class_names_path, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.1,
                draw_result=True):
        # sail.set_print_flag(True)
        # load bmodel
        self.draw_result = draw_result
        self.net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
        self.handle = self.net.get_handle()

        # get model info
        self.graph_name = self.net.get_graph_names()[0]

        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(
            self.graph_name, self.input_name)
        self.input_w = int(self.input_shape[-1])
        self.input_h = int(self.input_shape[-2])
        self.input_shapes = {self.input_name: self.input_shape}
        self.input_dtype = self.net.get_input_dtype(
            self.graph_name, self.input_name)

        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)

        self.input = sail.Tensor(
            self.handle, self.input_shape, self.input_dtype, True, True)
        self.input_tensors = {self.input_name: self.input}

        self.output_name = self.net.get_output_names(self.graph_name)[0]

        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_scale  = self.net.get_output_scale(self.graph_name, self.output_name)

        self.output = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True, True)

        self.output_tensors = {self.output_name: self.output}

        is_fp32 = (self.input_dtype == sail.Dtype.BM_FLOAT32)
        # get handle to create input and output tensors
        self.bmcv_converter = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv_converter.get_bm_image_data_format(self.input_dtype)

        # bgr normalization
        self.ab = [x * self.input_scale for x in [0.003921568627451, 0, 0.003921568627451, 0, 0.003921568627451, 0]]

        # generate anchor
        self.nl = 3
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62,
                                              45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(
            anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])

        # post-process threshold
        scalethreshold = 1.0 if is_fp32 else 0.9
        self.confThreshold = confThreshold * scalethreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold * scalethreshold
        self.ration = 1

        with open(class_names_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        print("input img_dtype:{}, input scale: {}, output scale: {} ".format(
            self.img_dtype, self.input_scale, self.output_scale))

        ## preprocess;infer;postprocess pipeline
        self.preprocess_queue = queue.Queue(QUEUE_SIZE)
        self.infer_queue = queue.Queue(QUEUE_SIZE)
        self.postprocess_queue = queue.Queue(QUEUE_SIZE)
        self.result_queue = queue.Queue(QUEUE_SIZE)

        preprocess_thread = threading.Thread(target = self.preprocess_with_bmcv, args = ())
        preprocess_thread.setDaemon(True)
        preprocess_thread.start()

        infer_thread = threading.Thread(target = self.predict, args = ())
        infer_thread.setDaemon(True)
        infer_thread.start()

        postprocess_thread = threading.Thread(target = self.postprocess, args = ())
        postprocess_thread.setDaemon(True)
        postprocess_thread.start()

        self.i = 0

    def compute_IOU(self, rec1, rec2):
        """
        计算两个矩形框的交并比。
        :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
        :param rec2: (x0,y0,x1,y1)
        :return: 交并比IOU.
        """
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])
        # 两矩形无相交区域的情况
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        # 两矩形有相交区域的情况
        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * \
                (right_column_min - left_column_max)
            return S_cross / S1

    def sigmoid(self, inx):  # 防止exp(-x)溢出
        indices_pos = np.nonzero(inx >= 0)
        indices_neg = np.nonzero(inx < 0)

        y = np.zeros_like(inx)
        y[indices_pos] = 1 / (1 + np.exp(-inx[indices_pos]))
        y[indices_neg] = np.exp(inx[indices_neg]) / \
            (1 + np.exp(inx[indices_neg]))

        return y
    
    def numpy_to_bmImage(self, img):
        h,w,c = img.shape
        # HWC->CHW
        img = np.transpose(img, [2, 0, 1])
        # CHW -> NCHW
        img = np.expand_dims(img, axis=0)
        # numpy -> sail.Tensor 
        sail_tensor = sail.Tensor(self.handle, img)
        # init bm_image
        bm_image = sail.BMImage(self.handle, h, w,
                                           sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        # sail.Tensor -> bm_image
        self.bmcv_converter.tensor_to_bm_image(sail_tensor, bm_image, bgr2rgb=True)
        return bm_image

    def preprocess_with_bmcv(self):
        while True:
            frame_id, t1, img = self.preprocess_queue.get()
            img = self.numpy_to_bmImage(img)
            img_w = img.width()
            img_h = img.height()
            # Calculate widht and height and paddings
            r_w = self.input_w / img_w
            r_h = self.input_h / img_h
            if r_h > r_w:
                tw = self.input_w
                th = int(r_w * img_h)
                tx1 = tx2 = 0
                ty1 = 0
                ty2 = self.input_h - th - ty1
                self.ration = r_w
            else:
                tw = int(r_h * img_w)
                th = self.input_h
                tx1 = 0
                tx2 = self.input_w - tw - tx1
                ty1 = ty2 = 0
                self.ration = r_h

            attr = sail.PaddingAtrr()
            attr.set_stx(tx1)
            attr.set_sty(ty1)
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)

            # preprocess 
            # padded_img_bgr = self.bmcv_converter.vpp_resize_padding(
            #    img, self.input_shape[2], self.input_shape[3], attr)
            
            padded_img_rgb = self.bmcv_converter.resize(img,
                                self.input_shape[3], self.input_shape[2])

            padded_img_rgb_norm = sail.BMImage(self.handle, self.input_shape[2], self.input_shape[3],
                                            sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
            self.bmcv_converter.convert_to(
                padded_img_rgb, padded_img_rgb_norm, ((self.ab[0], self.ab[1]), \
                                            (self.ab[2], self.ab[3]), \
                                            (self.ab[4], self.ab[5])))
            self.infer_queue.put((frame_id, t1, padded_img_rgb_norm))
        return 

    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype("float")

    def predict(self):
        while True:
            # print(self.infer_queue.qsize())
            frame_id, t1, data = self.infer_queue.get()
            # print(data)
            z = []  # inference output
            # print("use decoder data as input")
            self.bmcv_converter.bm_image_to_tensor(data, self.input)
            self.net.process(self.graph_name, self.input_tensors,
                                self.input_shapes, self.output_tensors)
            output = self.output.asnumpy(self.output_shape) * self.output_scale
            self.postprocess_queue.put((frame_id, t1, output))
        return

    def postprocess(self):
        while True:
            frame_id, t1, outs = self.postprocess_queue.get()
            # Scan through all the bounding boxes output from the network and keep only the
            # ones with high confidence scores. Assign the box's class label as the class with the highest score.
            classIds = []
            confidences = []
            boxes = []
            for out in outs:
                out = out[out[:, 4] > self.objThreshold, :]
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > self.confThreshold:
                        center_x = int(detection[0])
                        center_y = int(detection[1])
                        width = int(detection[2])
                        height = int(detection[3])

                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence)*detection[4])
                        boxes.append([left, top, width, height])
            # Perform nms to eliminate redundant overlapping boxes with lower confidences.
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, self.confThreshold, self.nmsThreshold)
            self.result_queue.put((frame_id, t1, indices, boxes, confidences, classIds))
        return 

    def inference(self, frame):
        if not isinstance(frame, type(None)):
            self.i += 1
            frame_id = self.i
            t1 = time.time()
            res_frame = frame
            self.preprocess_queue.put((frame_id,t1,frame))
            if self.result_queue.empty():
                print("======= skip, ", self.i)
                return frame, None
            frame_id, t1, indices, boxes, confidences, classIds = self.result_queue.get()
            t2 = time.time()
            #print("input_id: {}, result_id: {}, time: {}".format(self.i, frame_id, t2 - t1))
            if len(boxes) == 0:
                return res_frame, (frame_id,None)
            # opencv return (n,1) or (n, ) in different version
            if not isinstance(indices,tuple) and len(indices.shape) == 2:
                indices = indices.squeeze(1)

            res = []
            ratio = 1
            for i in indices:
                box = boxes[i]
                left = int((box[0]) / ratio)
                top = int((box[1]) / ratio)
                right = int((box[2] + box[0])/ratio)
                bottom = int((box[1] + box[3])/ratio)
                width = right - left
                height = bottom - top

                res.append({'det_box': [left, top, width, height],
                           "conf": confidences[i], "classId": classIds[i]})
                if self.draw_result:
                    res_frame = self.drawPred(res_frame, classIds[i], confidences[i], round(
                        left), round(top), round(right), round(bottom))
            return res_frame, (frame_id, res)


    def postprocess_np(self, outs, max_wh=7680):
        bs = outs.shape[0]
        output = [np.zeros((0, 6))] * bs
        xc = outs[..., 4] > self.confThreshold
        for xi, x in enumerate(outs):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            conf = x[:, 5:].max(1)
            j = x[:, 5:].argmax(1)
            x = np.concatenate((box, conf.reshape(-1, 1), j.reshape(-1, 1)), 1)[conf > self.confThreshold]
            c = x[:, 5:6] * max_wh  # classes
            boxes = x[:, :4] + c.reshape(-1, 1)
            scores = x[:, 4]
            i = nms_np(boxes, scores, self.nmsThreshold)
            output[xi] = x[i]

        return output



    def drawPred(self, frame, classId, conf, left, top, right, bottom):

        colors = (_COLORS[classId] * 255).astype(np.uint8).tolist()

        print("classid=%d, class=%s, conf=%f, (%d,%d,%d,%d)" %
              (classId, self.classes[classId], conf, left, top, right, bottom))
              
        # draw bboxes
        cv2.rectangle(frame, (left, top), (right, bottom),
                      colors, thickness=4)


        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[classId], thickness=2)
        return frame


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Demo of YOLOv5 with preprocess by BMCV")

    parser.add_argument('--bmodel',
                        type=str,
                        default="../data/models/yolov5s_640_coco_v6.1_1output_fp32_1b.bmodel",
                        required=False,
                        help='bmodel file path.')

    parser.add_argument('--labels',
                        type=str,
                        default="../data/coco.names",
                        required=False,
                        help='labels txt file path.')

    parser.add_argument('--input',
                        type=str,
                        default="../data/images/bus.jpg",
                        required=False,
                        help='input pic/video file path.')

    parser.add_argument('--tpu_id',
                        default=0,
                        type=int,
                        required=False,
                        help='tpu dev id(0,1,2,...).')

    parser.add_argument("--conf",
                        default=0.5,
                        type=float,
                        help="test conf threshold.")

    parser.add_argument("--nms",
                        default=0.5,
                        type=float,
                        help="test nms threshold.")

    parser.add_argument("--obj",
                        default=0.1,
                        type=float,
                        help="test obj conf.")

    parser.add_argument('--use_np_file_as_input',
                        default=False,
                        type=bool,
                        required=False,
                        help="whether use dumped numpy file as input.")

    opt = parser.parse_args()

    save_path = os.path.join(
        save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )

    if opt.use_np_file_as_input:
        save_path = save_path + "_numpy"

    os.makedirs(save_path, exist_ok=True)

    yolov5 = YOLOV5_Detector(bmodel_path=opt.bmodel,
                             tpu_id=opt.tpu_id,
                             class_names_path=opt.labels,
                             confThreshold=opt.conf,
                             nmsThreshold=opt.nms,
                             objThreshold=opt.obj)

    frame = cv2.imread(opt.input)

    print("processing file: {}".format(opt.input))

    if frame is not None:  # is picture file

        # decode
        decoder = sail.Decoder(opt.input, True, 0)

        input_bmimg = sail.BMImage()
        ret = decoder.read(yolov5.handle, input_bmimg)
        if ret:
            print("decode error\n")
            exit(-1)

        result_image = yolov5.inference_center_np(input_bmimg, opt.use_np_file_as_input)

        yolov5.bmcv.imwrite(os.path.join(save_path, "test_output.jpg"), result_image)

    else:  # is video file

        decoder = sail.Decoder(opt.input, True, 0)

        if decoder.is_opened():

            print("create decoder success")
            input_bmimg = sail.BMImage()
            id = 0

            while True:
                
                print("123")

                ret = decoder.read(yolov5.handle, input_bmimg)
                if ret:
                    print("decoder error\n")
                    break

                result_image = yolov5.inference_center(input_bmimg, use_np_file_as_input=False)

                yolov5.bmcv.imwrite(os.path.join(save_path, str(id) + ".jpg"), result_image)

                id += 1

            print("stream end or decoder error")

        else:
            print("failed to create decoder")

    print("===================================================")

