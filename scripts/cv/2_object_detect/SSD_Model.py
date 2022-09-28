import cv2
import sys
import os
import argparse
import json
import numpy as np
import sophon.sail as sail
import time

class PreProcessor:
    """ Preprocessing class.
    """

    def __init__(self, bmcv, scale):
        """ Constructor.
        """
        self.bmcv = bmcv
        self.ab = [x * scale for x in [1, -123, 1, -117, 1, -104]]

    def process(self, input, output):
        """ Execution function of preprocessing.
        Args:
          input: sail.BMImage, input image
          output: sail.BMImage, output data

        Returns:
          None
        """
        tmp = self.bmcv.vpp_resize(input, 300, 300)
        self.bmcv.convert_to(tmp, output, ((self.ab[0], self.ab[1]),
                                           (self.ab[2], self.ab[3]),
                                           (self.ab[4], self.ab[5])))


class PostProcessor:
    """ Postprocessing class.
    """

    def __init__(self, threshold):
        """ Constructor.
        """
        self.threshold = threshold

    def process(self, data, img_w, img_h):
        """ Execution function of postprocessing.
        Args:
          data: Inference output
          img_w: Image width
          img_h: Imgae height

        Returns:
          Detected boxes.
        """
        data = data.reshape((data.shape[2], data.shape[3]))
        ret = []
        for proposal in data:
            if proposal[2] < self.threshold:
                continue
            ret.append([
                int(proposal[1]),           # class id
                proposal[2],                # score
                int(proposal[3] * img_w),   # x0
                int(proposal[4] * img_h),   # x1
                int(proposal[5] * img_w),   # y0
                int(proposal[6] * img_h)])  # y1
        return ret

    def get_reference(self, compare_path):
        """ Get correct result from given file.
        Args:
          compare_path: Path to correct result file
        Returns:
          Correct result.
        """
        if compare_path:
            with open(compare_path, 'r') as f:
                reference = json.load(f)
                return reference["boxes"]
        return None

    def compare(self, reference, result, loop_id):
        """ Compare result.
        Args:
          reference: Correct result
          result: Output result
          loop_id: Loop iterator number

        Returns:
          True for success and False for failure
        """
        if not reference:
            #print("No verify_files file or verify_files err.")
            return True
        if loop_id > 0:
            return True
        data = []
        for line in result:
            cp_line = line.copy()
            cp_line[1] = "{:.8f}".format(cp_line[1])
            data.append(cp_line)
        if len(data) != len(reference):
            message = "Expected deteted number is {}, but detected {}!"
            print(message.format(len(reference), len(data)))
            return False
        ret = True
        message = "Category: {}, Score: {}, Box: [{}, {}, {}, {}]"
        fail_info = "Compare failed! Expect: " + message
        ret_info = "Result Box: " + message
        for i in range(len(data)):
            box = data[i]
            ref = reference[i]
            if box != ref:
                print(fail_info.format(ref[0], float(ref[1]), ref[2],
                                       ref[3], ref[4], ref[5]))
                print(ret_info.format(box[0], float(box[1]), box[2],
                                      box[3], box[4], box[5]))
                ret = False
        return ret


class BmodelEngine(object):
    tpu_id = 0
    i=0
    def __init__(self,bmodel_path, labels_path = None,single_w = 640,single_h=480):
        self.single_w=single_w
        self.single_h=single_h
        self.bmodel_path = bmodel_path
        """ Load a bmodel and do inference.
        Args:
        bmodel_path: Path to bmodel
        file_path: Path to input file
        加载模型
        """
        # sail.set_print_flag(True)
        # init Engine
        self.engine = sail.Engine(self.bmodel_path, self.tpu_id, sail.IOMode.DEVIO)
        # load bmodel without builtin input and output tensors
        #self.engine.load(self.bmodel_path)
        # get model info
        # only one model loaded for this engine
        # only one input tensor and only one output tensor in this graph
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]
        self.input_shape = [1, 3, 300, 300]
        self.input_shapes = {self.input_name: self.input_shape}
        output_shape = [1, 1, 200, 7]
        input_dtype  = self.engine.get_input_dtype(self.graph_name, self.input_name)
        output_dtype = self.engine.get_output_dtype(self.graph_name, self.output_name)
        is_fp32 = (input_dtype == sail.Dtype.BM_FLOAT32)
        # get handle to create input and output tensors
        self.handle = self.engine.get_handle()
        #self.input  = sail.Tensor(self.handle,self.input_shape,input_dtype,True,True)
        self.input  = sail.Tensor(self.handle,self.input_shape,sail.Dtype.BM_UINT8,True,True)

        self.output = sail.Tensor(self.handle,output_shape,output_dtype, True,True)
        self.input_tensors  = {self.input_name: self.input}
        self.output_tensors = {self.output_name: self.output}
        # set io_mode
        self.engine.set_io_mode(self.graph_name, sail.IOMode.SYSO)
        # init bmcv for preprocess
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(input_dtype)
        # init preprocessor and postprocessor
        scale = self.engine.get_input_scale(self.graph_name, self.input_name)
        self.preprocessor = PreProcessor(self.bmcv, scale)
        threshold = 0.59 if is_fp32 else 0.52
        self.postprocessor = PostProcessor(threshold)
        self.reference = self.postprocessor.get_reference("")
        
        self.labels = self._read_labels(labels_path)
        self.dets = []

    def bmimage_to_cvmat(self, bmimage):
        result_tensor = self.bmcv.bm_image_to_tensor(bmimage)
        result_numpy  = result_tensor.asnumpy()
        np_array_temp = result_numpy[0]
        np_array_t    = np.transpose(np_array_temp, [1, 2, 0])
        image         = np.array(np_array_t, dtype=np.uint8)
        b,g,r = cv2.split(image)
        image = cv2.merge([b,g,r])
        return image
    
    # return detect result,
    #        
    def get_results(self, img0):
        self.i+=1
        ori_height, ori_width = img0.shape[:2]
        if self.i % 3 == 0:
            #开始模型推理
            # 1.preprocess
            img1 = cv2.resize(img0, (300, 300))
            linear_convert = [(1,-123),(1, -117),(1, -104)]
            for i in range(3):
                img1[i] = linear_convert[i][0] * img1[i] + linear_convert[i][1]
            img1 = np.transpose(img1, [2,0,1])
            self.input.update_data(np.expand_dims(img1,axis=0))

            # 2.inference
            self.engine.process(self.graph_name, self.input_tensors,self.input_shapes,self.output_tensors)
           
            # 3.postprocess
            real_output_shape = self.engine.get_output_shape( self.graph_name, self.output_name)
            out =  self.output.asnumpy(real_output_shape)
            self.dets = self.postprocessor.process(out, ori_width, ori_height)
        return self.dets

    def get_inference_img(self,img0):
        
        self.get_results(img0)
        # 4. draw result on bmimage
        if self.postprocessor.compare(self.reference, self.dets, self.i):
            for (class_id, score, x0, y0, x1, y1) in self.dets:
                # import pdb;pdb.set_trace()
                message = "{0},score:{1}".format(self.labels[class_id], score)
                #print(message.format(self.i , self.tpu_id, class_id, score, x0, y0, x1, y1))
                cv2.putText(img0,message, (x0,y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                cv2.rectangle(img0, (x0, y0), (x1, y1), (0,0,255), 2)

        # msg = "o_w=%d o_=%d"%(ori_width, ori_height)
        # cv2.putText(img0, msg, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        #print("post time :", time.time() - end)
        return img0
    
    @staticmethod
    def _read_labels(labels_txt):
        cls_id = 0
        labels = {}
        with open(labels_txt, 'r') as f:
            while True:
                line = f.readline().strip()
                if line == "": break
                labels[cls_id] = line
                cls_id += 1
        return labels


        
