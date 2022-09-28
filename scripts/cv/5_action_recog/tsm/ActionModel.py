import cv2
import numpy as np

import sophon.sail as sail
catigories = [
        "Doing other things",  # 0           ####
        "Drumming Fingers",  # 1
        "No gesture",  # 2
        "Pulling Hand In",  # 3
        "Pulling Two Fingers In",  # 4
        "Pushing Hand Away",  # 5
        "Pushing Two Fingers Away",  # 6
        "Rolling Hand Backward",  # 7         ####
        "Rolling Hand Forward",  # 8          ####
        "Shaking Hand",  # 9                  ###
        "Sliding Two Fingers Down",  # 10     ###
        "Sliding Two Fingers Left",  # 11     ###
        "Sliding Two Fingers Right",  # 12    ###
        "Sliding Two Fingers Up",  # 13       ###
        "Stop Sign",  # 14
        "Swiping Down",  # 15
        "Swiping Left",  # 16
        "Swiping Right",  # 17
        "Swiping Up",  # 18
        "Thumb Down",  # 19
        "Thumb Up",  # 20
        "Turning Hand Clockwise",  # 21        ###
        "Turning Hand Counterclockwise",  # 22 ###
        "Zooming In With Full Hand",  # 23     ###
        "Zooming In With Two Fingers",  # 24   ###
        "Zooming Out With Full Hand",  # 25    ###
        "Zooming Out With Two Fingers"  # 26   ###
        ]
class PreProcesser(object):
    """ 
    """
    def __init__(self, mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, frame):
        # BGR => RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self._scale(frame)
        frame = self._center_crop(frame)
        frame = self._tranpose(frame)
        frame = self._normalize(frame)
        frame = np.expand_dims(frame, axis=0)
        return frame

    def _scale(self, frame, size = 256):
        """rescale img to given 'size'.'size' will be the size of the smaller edge.
        For example, if height > width, image will rescaled to (size * height/width, size)
        """
        img_h, img_w, img_c = frame.shape
        if img_h > img_w:
            out_w = size
            out_h = img_h / img_w * size
        else:
            out_h = size
            out_w = img_w / img_h * size
        frame = cv2.resize(frame, (int(out_w), int(out_h)), cv2.INTER_LINEAR)
        return frame
    
    def _center_crop(self, frame, size = 224):
        img_h, img_w, img_c = frame.shape
        h_delate = (img_h - size) // 2
        w_delate = (img_w - size) // 2
        crop_frame = frame[h_delate : h_delate + size ,
                           w_delate : w_delate + size,
                           :]
        return crop_frame

    def _tranpose(self, frame):
        frame = np.transpose(frame, [2, 0, 1])
        return frame
    
    def _normalize(self, frame):
        for i in range(frame.shape[0]):
            frame[i] = (frame[i] / 255.0 - self.mean[i]) / self.std[i]
        return frame


class PostProcessor(object):
    """fix result
    """
    def __init__(self):
        self.history = [2]
        self.max_hist_len = 20 # max history buffer
        self.i = 0
        self.idx = 2

    def __call__(self, idx_):
        
        # mask out illegal action 
        if idx_ in [7, 8, 21, 22, 3, 23, 24, 25, 26]:
            idx_ = self.history[-1]

        # use only single no action class
        if idx_ == 0:
            idx_ = 2
        
        # merge class
        merge_class = {10: 15, 11: 16, 12: 17, 13: 18}
        if idx_ in [10, 11, 12, 13]:
            idx_ = merge_class[idx_]
        
        # history smoothing
        if idx_ != self.history[-1]:
            if not (self.history[-1] == self.history[-2]): #  and history[-2] == history[-3]):
                idx_ = self.history[-1]

        self.history.append(idx_)
        self.history = self.history[-self.max_hist_len:]
        
        return idx_



class ActRecognizer(object):
    """Act Recognitation
    """
    def __init__(self, bmodel_path, tpu_id = 0):
        self.preprocessor  = PreProcesser()
        self.postprocessor = PostProcessor()
        self.engine = TSM(bmodel_path, tpu_id)
        self.interval = 2
        self.i = 0
        self.label = catigories[2]
        print("create ActRecognizer !!!!")

    def inference(self, frame):
        left , top = 91, 11
        right, bottom = 549, 469 
        cv2.rectangle(frame, (left, top), (right, bottom),(255,0,0), 4)
        self.i += 1
        if self.i % self.interval == 0:
            input = self.preprocessor(frame)
            idx   = self.engine(input)
            idx2  = self.postprocessor(idx)
            self.label = catigories[idx2]
        cv2.putText(frame, self.label, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        return frame

class TSM(object):
    def __init__(self, bmodel_path, tpu_id):
        # create infer engine, and set mode SYSIO
        self.net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
        # get engine s' handle for other usage 
        self.handle = self.net.get_handle()
        
        # get engine name
        self.graph_name = self.net.get_graph_names()[0]
        # get engine's input name, and init input tensor
        self.input_names = self.net.get_input_names(self.graph_name)
        self.inputs = {}
        for iname in self.input_names:
            in_shape = self.net.get_input_shape(self.graph_name, iname)
            input_dtype = self.net.get_input_dtype(self.graph_name, iname)
            input_tensor = sail.Tensor(self.handle, in_shape, input_dtype, True, True)
            self.inputs[iname] = input_tensor

        # get engine's output name
        self.outputs = {}
        self.output_names = self.net.get_output_names(self.graph_name)
        for oname in self.output_names:
            o_shape = self.net.get_output_shape(self.graph_name, oname)
            o_dtype = self.net.get_output_dtype(self.graph_name, oname)
            o_tensor = sail.Tensor(self.handle, o_shape, o_dtype, True, True)
            self.outputs[oname] = o_tensor

        # history fea
        self.history_fea = []

    def __call__(self, data):
        """get processed data and return result
        """
        # update input data
        self.inputs["input"].update_data(data)
        # infer
        self.net.process(self.graph_name, self.inputs, self.outputs)
        # update buffer data
        for i in range(1, len(self.input_names)):
            iname = self.input_names[i]
            oname = self.output_names[i]
            self.inputs[iname].update_data(self.outputs[oname].asnumpy())
        out = self.outputs["cls"].asnumpy()
        
        # update history fea
        self.history_fea.append(out)
        self.history_fea = self.history_fea[-12:]
        # return res
        avg_logit = sum(self.history_fea)
        idx_ = np.argmax(avg_logit, axis = 1)[0]
        return idx_


    def inference(self, frame):
        input = self.preprocessor(frame)

if __name__ == "__main__":
    bmodel_path = "/home/linaro/workspace/robot_ws/src/sophon_robot/data/cv/5_action_recog/tsm_mobilenet_v2_jester_1x224x244.bmodel"
    tpu_id = 0
    infer = TSM(bmodel_path, tpu_id)