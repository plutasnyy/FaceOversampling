from __future__ import print_function

import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# noinspection DuplicatedCode
from retina_data import cfg_mnet, cfg_re50
from retina_layers.functions.prior_box import PriorBox
from retina_models.retinaface import RetinaFace
from retina_utils.box_utils import decode, decode_landm
from retina_utils.nms.py_cpu_nms import py_cpu_nms


class FaceDetector:
    def __init__(self, weights='models/retina_face/Resnet50_Final.pth', network='resnet50', keep_top_k=1):
        torch.set_grad_enabled(False)
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = keep_top_k
        self.cfg = None

        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50

        net = RetinaFace(cfg=self.cfg, phase='test')
        net = self.load_model(net, weights, load_to_cpu=False)
        net.eval()

        cudnn.benchmark = True
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cuda")
        self.net = net.to(self.device)

    def predict(self, img_raw):
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.02)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        return dets, landms

    def load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    @staticmethod
    def remove_prefix(state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    @staticmethod
    def check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    @staticmethod
    def save_img_with_dets(img_raw, dets, landmarks):
        for b, l in zip(dets, landmarks):
            if b[4] < 0.6:
                continue
            text = "{:.4f}".format(b[4])
            b, l = list(map(int, b)), list(map(int, l))

            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.putText(img_raw, text, (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            cv2.circle(img_raw, (l[0], l[1]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (l[2], l[3]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (l[4], l[5]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (l[6], l[7]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (l[8], l[9]), 1, (255, 0, 0), 4)

        cv2.imwrite('Title2.png', img_raw)


if __name__ == '__main__':
    face_detector = FaceDetector()

    image_path = '/home/plutasnyy/git/pbp/data/ffhq/68709.png'
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

    dets, landmarks = face_detector.predict(img_raw)
    face_detector.save_img_with_dets(img_raw, dets, landmarks)
