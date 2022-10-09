import cv2
import numpy as np

import sophon.sail as sail

from transform import TopDownGetBboxCenterScale, TopDownAffine, \
                        keypoints_from_heatmaps

from show_results import imshow_keypoints

class PreProcesser(object):
    """ 
    """
    def __init__(self, mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.TopDownGetBboxCenterScale = TopDownGetBboxCenterScale(padding = 1.25)
        self.TopDownAffine = TopDownAffine()

    def __call__(self, frame, bboxes):
        # BGR => RGB
        p_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for i,box in enumerate(bboxes.bboxes): # bboxes.bboxes
            data = {
                    "img" : p_frame,
                    "bbox": [box.left_top_x, box.left_top_y,
                            box.width, box.height],
                    # "bbox" : box,
                    "bbox_score": 1,
                    "bbox_id": i,
                    "rotation": 0,
                    "ann_info":{
                        "image_size": np.array([192,256]),
                        "num_joints": 17,
                        'flip_pairs':[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
                        }
                    }
            data = self.TopDownGetBboxCenterScale(data)
            data = self.TopDownAffine(data)
            data = self._tranpose(data, div_255 = True)
            data = self._normalize(data)
            data['img'] = np.expand_dims(data['img'], axis=0)
            return data

    def _tranpose(self, data, div_255 = True):
        data['img'] = np.transpose(data['img'], [2, 0, 1])
        if div_255:
            data["img"] = data["img"] / 255
        return data

    def _normalize(self, data):
        for i in range(data['img'].shape[0]):
            data['img'][i] = (data['img'][i] - self.mean[i]) / self.std[i]
        return data



class PostProcessor(object):
    """decode keypoints from heatmaps
    ## preds(np.array)'s shape: N, K, 3.
                    N - the num of person
                    K - the num of keypoints
                    3 - x, y, score
    """
    def __init__(self,):
        pass

    def __call__(self, heatmap, center, scale):
        preds, maxvals = keypoints_from_heatmaps(
                            heatmap, center, scale)
        pred = np.concatenate((preds, maxvals), axis = 2)
        return pred

class Kpt(object):
    """Kpt 
    """
    def __init__(self, bmodel_path, tpu_id = 0):
        self.preprocessor  = PreProcesser()
        self.postprocessor = PostProcessor()
        self.engine = TopDownPose(bmodel_path, tpu_id)
        self.interval = 1
        self.i = -1
        self.preds = None
        print("create person keypoint proposer!!!!")

    def inference(self, frame, bboxes):
        self.i += 1
        if bboxes.num_object == 0:
            return frame
        if self.i % self.interval == 0:
            processed_data = self.preprocessor(frame, bboxes)
            heatmaps  = self.engine(processed_data["img"])
            self.preds  = self.postprocessor( heatmaps,
                                        processed_data["center"],
                                        processed_data["scale"])
        elif self.i % self.interval == self.interval - 1:
            self.preds = None
        print(self.preds)
        imshow_keypoints(frame, self.preds)
        return frame

class TopDownPose(object):
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

    def __call__(self, data):
        """get processed data and return result
            data : 
        """
        # update input data
        self.inputs["input.1"].update_data(data)
        # infer
        self.net.process(self.graph_name, self.inputs, self.outputs)

        out = self.outputs["520"].asnumpy()

        return out

if __name__ == "__main__":
    bmodel_path = "/home/linaro/workspace/robot_ws/src/sophon_robot/data/cv/5_action_recog/res50_coco_256x192-ec54d7f3_20200709.bmodel"
    img_path = "/home/linaro/workspace/robot_ws/src/sophon_robot/data/cv/5_action_recog/000000000785.jpg"
    frame = cv2.imread(img_path)
    kptor = Kpt(bmodel_path)
    kptor.inference(frame, [[280, 44, 218, 346]])