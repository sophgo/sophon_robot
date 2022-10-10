
import numpy as np
import base64
import os
import cv2
import torch

import sophon.sail as sail 
import numpy as np

from utils.timer import Timer
from utils.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.align import warp_and_crop_face


_t = {'forward_pass': Timer(), 'misc': Timer()}

cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


def preprocess(img,input_size):
    height = img.shape[0]
    width = img.shape[1]
    size = max(width, height)
    img_padding = cv2.copyMakeBorder(img, 0, size - height, 0, size - width, cv2.BORDER_CONSTANT, (0, 0, 0))
    image_array = cv2.resize(img_padding,(input_size,input_size))
    image_array = image_array.transpose(2, 0, 1)
    tensor = np.expand_dims(image_array, axis=0)
    tensor = np.ascontiguousarray(tensor)
    return tensor,height,width

def facepreprocess(img):

    image_array = cv2.resize(img,(112,112))
    image_array = image_array.transpose(2, 0, 1)
    tensor = np.expand_dims(image_array, axis=0)
    tensor = np.ascontiguousarray(tensor)
    return tensor

class face(object):
    def __init__(self,faceDP,faceRP, known_people_path=""):
        self.face_net = sail.Engine(faceRP,  0 , sail.IOMode.SYSIO)
        self.face_graph_name = self.face_net.get_graph_names()[0] 
        self.face_input_names = self.face_net.get_input_names(self.face_graph_name)[0] 

        self.net = sail.Engine(faceDP,  0 , sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0] 
        self.input_names = self.net.get_input_names(self.graph_name)[0]  
        print('bmodel init sucess!')
        self.resize = 640
        self.img_input = 640
        self.confidence_threshold = 0.02
        self.nms_threshold = 0.4
        self.vis_thres = 0.5
        self.is_same_person_conf = 0.8

        self.known_people = {}
        pics = os.listdir(known_people_path)
        for pic in pics:
            pic_path = os.path.join(known_people_path, pic)
            img = cv2.imread(pic_path)
            feas, _,__ = self.getFeature(img)
            self.known_people[pic.split(".")[0]] = feas

    def inference(self, img):
        feas, img_draw, det  = self.getFeature(img)
        cos_score = 0
        person = None
        if len(feas) == 0:
            return img_draw
        for person_name, know_fea in self.known_people.items():
            score = self.getScoreCos(feas[0], know_fea[0])
            if score > cos_score :
                cos_score, person = score, person_name 
        if person is not None and cos_score > self.is_same_person_conf:
            print(person, cos_score)
            cv2.putText(img_draw, person, (det[0], det[1]),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        
        return img_draw

    def getFaceFeature(self,img,pts):
        faceimg = warp_and_crop_face(img,pts)
        output = self.face_net.process(self.face_graph_name, {self.face_input_names: faceimg})
        return output

    # get every face's feature, get only one person'face 
    def getFeature(self, img):
        dets,ldms,imgdraw,textPos = self.detect(img)
        _t['forward_pass'].tic()
        choosen_face_ldm, score, choosen_face_det = None, -1, None
        for det,ldm in zip(dets,ldms):
            if det[4] < self.vis_thres:
                continue
            if det[4] > score:
                ldm.resize([5,2])
                score = det[4]
                choosen_face_det = det
                choosen_face_ldm = ldm

        output=[]
        if choosen_face_ldm is not None:
            faceimg = warp_and_crop_face(img, choosen_face_ldm)
            faceinput = facepreprocess(faceimg)
            feature_nmodel = self.face_net.process(self.face_graph_name, {self.face_input_names: faceinput})
            output.append(feature_nmodel['24'])
            self.draw_box_and_ldmk(imgdraw, choosen_face_det, choosen_face_ldm)

        _t['forward_pass'].toc()
        print('face_recog: {} faces, forward_pass_time: {:.4f}ms '.format(len(dets),1000 * _t['forward_pass'].average_time))
        return output,imgdraw,choosen_face_det 

    def detect(self,img_src):
        img,im_height,im_width = preprocess(img_src,self.img_input)
        scale = max(im_height,im_width)/self.img_input
        # print(input_array.shape)
        # while(1):
        _t['forward_pass'].tic()
        output = self.net.process(self.graph_name, {self.input_names: img})
        landms = torch.Tensor(output['111'])
        loc = torch.Tensor(output['87'])
        conf = torch.Tensor(output['116'])
        _t['forward_pass'].toc()
        priorbox = PriorBox(cfg, image_size=(self.img_input, self.img_input))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        scale_box = torch.Tensor([img_src.shape[1], img_src.shape[0], img_src.shape[1], img_src.shape[0]])
        boxes = boxes * scale *self.img_input
        boxes = boxes.numpy()
        scores = conf.squeeze(0).data.numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        dotldms =  torch.Tensor([img_src.shape[1],img_src.shape[0],img_src.shape[1],img_src.shape[0],img_src.shape[1],img_src.shape[0],img_src.shape[1],img_src.shape[0],img_src.shape[1],img_src.shape[0]])

        landms = landms * scale * self.img_input
        landms = landms.numpy()
        print('face_detect forward_pass_time: {:.4f}ms '.format(1000 * _t['forward_pass'].average_time))
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:args.top_k]
        order = scores.argsort()[::-1]
        order=np.ascontiguousarray(order, dtype=np.int)
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)

        dets = dets[keep, :]
        landms = landms[keep]
        dets_ = np.concatenate((dets, landms), axis=1)
        textPos=[]
        for b in dets_:
            if b[4] < self.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # don't draw on img
            # cv2.rectangle(img_src, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            textPos.append([cx,cy])
            # cv2.putText(img_src, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # don't draw landms
            # # landms
            # cv2.circle(img_src, (b[5], b[6]), 1, (0, 0, 255), 4)
            # cv2.circle(img_src, (b[7], b[8]), 1, (0, 255, 255), 4)
            # cv2.circle(img_src, (b[9], b[10]), 1, (255, 0, 255), 4)
            # cv2.circle(img_src, (b[11], b[12]), 1, (0, 255, 0), 4)
            # cv2.circle(img_src, (b[13], b[14]), 1, (255, 0, 0), 4)
        detsR = []
        landmsR =[]
        for det,ldm in zip(dets,landms):
            if det[4] < self.vis_thres:
                continue
            detsR.append(det)
            landmsR.append(ldm)
        return detsR,landmsR,img_src,textPos

    def draw_box_and_ldmk(self, img_src, dets, ldmks):
        """
        dets: (list) [x,y,x,y]
        ldmks: (np.array) [[x,y], [x,y], [x,y], [x,y], [x,y]]
        """
        dets = list(map(int, dets))
        
        ldmks = ldmks.astype(int)
        cv2.rectangle(img_src, (dets[0], dets[1]), (dets[2], dets[3]), (0, 0, 255), 2)
        cv2.circle(img_src, (ldmks[0][0], ldmks[0][1]), 1, (0, 0, 255), 4)
        cv2.circle(img_src, (ldmks[1][0], ldmks[1][1]), 1, (0, 255, 255), 4)
        cv2.circle(img_src, (ldmks[2][0], ldmks[2][1]), 1, (255, 0, 255), 4)
        cv2.circle(img_src, (ldmks[3][0], ldmks[3][1]), 1, (0, 255, 0), 4)
        cv2.circle(img_src, (ldmks[4][0], ldmks[4][1]), 1, (255, 0, 0), 4)

    def getScoreCos(self,a,b):
        v1 = a[0]
        v2 = b[0]
        angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
        return angle

