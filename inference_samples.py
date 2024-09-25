# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from glob import glob
import os


import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from src.models.model import ESANet
import warnings
from torch.utils.data import DataLoader
from src import preprocessing
import rospy
from sensor_msgs.msg import Image
from src.args import ArgumentParserRGBDSegmentation
from src.models.model_one_modality import ESANetOneModality
from src.build_model import build_model
from src.prepare_data import prepare_data
from src.datasets import SUNRGBD

from tamlib.node_template import Node
from tamlib.cv_bridge import CvBridge


class IndoorSegmentation(Node):
    def __init__(self) -> None:
        super().__init__()
        self.n_classes = 37  # sunrgbd
        self.height = 480
        self.width = 640
        self.encoder = "resnet34"
        self.encoder_block = "NonBottleneck1D"
        self.pretrained_dir = "./trained_models/imagenet"
        self.pretrained_on_imagenet = False  # 事前学習モデルを利用するかどうか
        self.encoder_depth = None
        self.activation = "relu"
        self.encoder_decoder_fusion = 'add'
        self.context_module = 'ppm'
        self.nr_decoder_blocks = [3]
        self.decoder_channels_mode = "decreasing"
        self.channels_decoder = 128
        self.fuse_depth_in_rgb_encoder = 'SE-add'
        self.upsampling = 'learned-3x3-zeropad'
        self.ckpt_path = './trained_models/sunrgbd/r34_NBt1D.pth'
        self.depth_scale = 1.0
        self.last_ckpt = ""
        self.pretrained_scenenet = ""

        if not self.pretrained_on_imagenet or self.last_ckpt or self.pretrained_scenenet != '':
            self.pretrained_on_imagenet = False
        else:
            self.pretrained_on_imagenet = True

        print(self.pretrained_on_imagenet)

        if 'decreasing' in self.decoder_channels_mode:
            if self.decoder_channels_mode == 'decreasing':
                channels_decoder = [512, 256, 128]

            warnings.warn('Argument --channels_decoder is ignored when '
                        '--decoder_chanels_mode decreasing is set.')
        else:
            channels_decoder = [self.channels_decoder] * 3

        if isinstance(self.nr_decoder_blocks, int):
            nr_decoder_blocks = [self.nr_decoder_blocks] * 3
        elif len(self.nr_decoder_blocks) == 1:
            nr_decoder_blocks = self.nr_decoder_blocks * 3
        else:
            nr_decoder_blocks = self.nr_decoder_blocks
            assert len(nr_decoder_blocks) == 3

        if self.encoder_depth in [None, 'None']:
            self.encoder_depth = self.encoder

        self.model = ESANet(
            height=self.height,
            width=self.width,
            num_classes=self.n_classes,
            pretrained_on_imagenet=self.pretrained_on_imagenet,
            pretrained_dir=self.pretrained_dir,
            encoder_rgb=self.encoder,
            encoder_depth=self.encoder_depth,
            encoder_block=self.encoder_block,
            activation=self.activation,
            encoder_decoder_fusion=self.encoder_decoder_fusion,
            context_module=self.context_module,
            nr_decoder_blocks=nr_decoder_blocks,
            channels_decoder=channels_decoder,
            fuse_depth_in_rgb_encoder=self.fuse_depth_in_rgb_encoder,
            upsampling=self.upsampling
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.checkpoint = torch.load(self.ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.checkpoint['state_dict'])

        self.model.eval()
        self.model.to(self.device)

        # sunrgbdの設定
        self.dataset_dir = None
        Dataset = SUNRGBD
        self.depth_mode = 'raw'
        self.with_input_orig = True
        dataset_kwargs = {}
        valid_set = 'test'

        self.train_data = Dataset(
            data_dir=self.dataset_dir,
            split='train',
            depth_mode=self.depth_mode,
            with_input_orig=self.with_input_orig,
            **dataset_kwargs
        )

        depth_stats = {'mean': self.train_data.depth_mean,
                       'std': self.train_data.depth_std}

        self.valid_preprocessor = preprocessing.get_preprocessor(
            height=self.height,
            width=self.width,
            depth_mean=depth_stats['mean'],
            depth_std=depth_stats['std'],
            depth_mode=self.depth_mode,
            phase='test'
        )

        self.valid_data = Dataset(
            data_dir=self.dataset_dir,
            split=valid_set,
            depth_mode=self.depth_mode,
            with_input_orig=self.with_input_orig,
            **dataset_kwargs
        )
        self.valid_data.preprocessor = self.valid_preprocessor
        # self.valid_loader = DataLoader(valid_data, bat
        # h_size=batch_size_valid, num_workers=args.workers, shuffle=False)

        # ros inter face
        p_rgb_topic = "/camera/rgb/image_raw"
        p_depth_topic = "/camera/depth/image_raw"

        self.msg_rgb = Image()
        self.msg_depth = Image()
        topics = {"msg_rgb": p_rgb_topic, "msg_depth": p_depth_topic}
        self.cv_bridge = CvBridge()
        self.sync_sub_register("rgbd", topics=topics, callback_func=self.run, delay=0.5)
        self.pub_register("indoor_seg", "/ESANet/result/image", Image, queue_size=1)

    def __del__(self):
        # return super().__del__()
        return

    def load_img(self, fp):
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def run(self, msg_rgb, msg_depth):
        img_rgb = self.cv_bridge.imgmsg_to_cv2(img_msg=msg_rgb, encoding="bgr8")
        img_depth = self.cv_bridge.imgmsg_to_cv2(img_msg=msg_depth)
        img_depth = img_depth.astype('float32')
        # img_depth = img_depth.astype('float32') * self.depth_scale

        h, w, _ = img_rgb.shape

        # preprocess sample
        sample = self.valid_preprocessor({'image': img_rgb, 'depth': img_depth})

        # add batch axis and copy to device
        image = sample['image'][None].to(self.device)
        depth = sample['depth'][None].to(self.device)

        # apply network
        pred = self.model(image, depth)
        pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy().squeeze().astype(np.uint8)

        # show result
        pred_colored = self.valid_data.color_label(pred, with_void=False)

        result_msg = self.cv_bridge.cv2_to_imgmsg(pred_colored)
        self.pub.indoor_seg.publish(result_msg)


if __name__ == '__main__':
    rospy.init_node("esanet_ros_noetic")

    p_loop_rate = 10
    loop_wait = rospy.Rate(p_loop_rate)

    cls = IndoorSegmentation()
    rospy.on_shutdown(cls.delete)

    while not rospy.is_shutdown():
        loop_wait.sleep()

# Namespace(activation='relu', aug_scale_max=1.4, aug_scale_min=1.0, batch_size=8, batch_size_valid=None, c_for_logarithmic_weighting=1.02, channels_decoder=128, ckpt_path='./trained_models/sunrgbd/r34_NBt1D.pth', class_weighting='median_frequency', context_module='ppm', dataset='sunrgbd', dataset_dir=None, debug=False, decoder_channels_mode='decreasing', depth_scale=1.0, encoder='resnet34', encoder_block='NonBottleneck1D', encoder_decoder_fusion='add', encoder_depth=None, epochs=500, finetune=None, freeze=0, fuse_depth_in_rgb_encoder='SE-add', he_init=False, height=480, last_ckpt='', lr=0.01, modality='rgbd', momentum=0.9, nr_decoder_blocks=[3], optimizer='SGD', pretrained_dir='./trained_models/imagenet', pretrained_on_imagenet=True, pretrained_scenenet='', raw_depth=True, results_dir='./results', upsampling='learned-3x3-zeropad', valid_full_res=False, weight_decay=0.0001, width=640, workers=8)