'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import sys
sys.path.append("../GPEN")
import os
import cv2
import glob
import time
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw
import __init_paths
from face_enhancement import FaceEnhancement
# from face_colorization import FaceColorization
# from face_inpainting import FaceInpainting
# from segmentation2face import Segmentation2Face


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GPEN-BFR-512', help='GPEN model')
parser.add_argument('--task', type=str, default='FaceEnhancement', help='task of GPEN model')
parser.add_argument('--key', type=str, default=None, help='key of GPEN model')
parser.add_argument('--in_size', type=int, default=512, help='in resolution of GPEN')
parser.add_argument('--out_size', type=int, default=None, help='out resolution of GPEN')
parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
parser.add_argument('--narrow', type=float, default=1, help='channel narrow scale')
parser.add_argument('--alpha', type=float, default=1, help='blending the results')
parser.add_argument('--use_sr', action='store_true', help='use sr or not')
parser.add_argument('--use_cuda', action='store_true', help='use cuda or not')
parser.add_argument('--save_face', action='store_true', help='save face or not')
parser.add_argument('--aligned', action='store_true', help='input are aligned faces or not')
parser.add_argument('--sr_model', type=str, default='realesrnet', help='SR model')
parser.add_argument('--sr_scale', type=int, default=2, help='SR scale')
parser.add_argument('--tile_size', type=int, default=0, help='tile size for SR to avoid OOM')
parser.add_argument('--indir', type=str, default='examples/imgs', help='input folder')
parser.add_argument('--outdir', type=str, default='results/outs-bfr', help='output folder')
parser.add_argument('--ext', type=str, default='.jpg', help='extension of output')
# args = parser.parse_args() 
# ------------------------------------------

gpen_args = parser.parse_args(args=[])
gpen_args.use_cuda = True


def gpen_enhance(img,faceenhancer):
    ## GPEN requires BGR images
    img_out, orig_faces, enhanced_faces = faceenhancer.process(img, aligned=False)
    
    ##img = cv2.resize(img, img_out.shape[:2][::-1])
    
    return img_out

faceenhancer = FaceEnhancement(gpen_args, base_dir='../GPEN',in_size=gpen_args.in_size, model=gpen_args.model, use_sr=gpen_args.use_sr, device='cuda' if gpen_args.use_cuda else 'cpu')
