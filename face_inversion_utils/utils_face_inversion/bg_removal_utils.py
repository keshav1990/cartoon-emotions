
import sys
import PIL 
# from hfgi_utils import tranform_img
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import numpy as np

import sys
sys.path.insert(1,'repositories_used/MODNet/')
from src.models.modnet import MODNet
# import torch
import torch.nn as nn


tranform_img = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])  


def find_matte(im, modnet):
    h = im.height
    w = im.width

    ref_size = 512
    # # unify image channels to 3
    # im = np.asarray(im)
    # if len(im.shape) == 2:
    #     im = im[:, :, None]
    # if im.shape[2] == 1:
    #     im = np.repeat(im, 3, axis=2)
    # elif im.shape[2] == 4:
    #     im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    # im = Image.fromarray(im)
    im = tranform_img(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    # Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(args.output_path, matte_name))
    # matte = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
    matte = PIL.Image.fromarray(((matte * 255).astype('uint8')), mode='L').resize((w,h))
    return matte

def map_matte(image, matte, bg_color):
    # calculate display resolution
    w, h = image.width, image.height
    rw, rh = 800, int(h * 800 / (3 * w))

    # obtain predicted foreground
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0:3]
    
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + np.full(image.shape, bg_color) * (1 - matte)

    # # combine image, foreground, and alpha into one line
    # combined = np.concatenate((image, foreground, matte * 255), axis=1)
    # combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))
    # foreground = PIL.Image.fromarray(np.uint8(foreground))
    return np.uint8(foreground),matte

def remove_bg(img, model, bg_color=[255,255,255],get_transparent=True):
    mat = find_matte(img, model)
    foreground,matte = map_matte(img, mat, bg_color)    
    
    if get_transparent==True:
        img_rgba = img.convert('RGBA')
        img_rgba = np.array(img_rgba)
        img_rgba[:,:,3] = np.array(mat)
        
        return foreground,img_rgba
    
    else:
        return foreground


def load_modnet(ckpt_path='modnet_photographic_portrait_matting.ckpt'):

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        print('------------------------------------')
        print('Loading modnet on cuda')
        print('------------------------------------')

        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        print('------------------------------------')
        print('Loading modnet on cpu')
        print('------------------------------------')
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    modnet.load_state_dict(weights)
    modnet.eval()

    return modnet
