
from utils.face_alignment_utils import align_face
import os

import sys
sys.path.append('../')
sys.path.append('repositories_used/HFGI/')
sys.path.append('../repositories_used/HFGI/')

print(os.listdir())


import numpy as np
import random
import torch
import torchvision.transforms as transforms
from repositories_used.HFGI.hfgi_repo_utils.common import tensor2im
from repositories_used.HFGI.models.psp import pSp  # we use the pSp framework to load the e4e encoder.

from argparse import Namespace

# (idx, edit_start, edit_end, strength, invert)
ganspace_directions = {

    # StyleGAN2 ffhq
    'frizzy_hair':             (31,  2,  6,  20, False),
    'background_blur':         (49,  6,  9,  20, False),
    'bald':                    (21,  2,  5,  20, False),
    'big_smile':               (19,  4,  5,  20, False),
    'caricature_smile':        (26,  3,  8,  13, False),
    'scary_eyes':              (33,  6,  8,  20, False),
    'curly_hair':              (47,  3,  6,  20, False),
    'dark_bg_shiny_hair':      (13,  8,  9,  20, False),
    'dark_hair_and_light_pos': (14,  8,  9,  20, False),
    'dark_hair':               (16,  8,  9,  20, False),
    'disgusted':               (43,  6,  8, -30, False),
    'displeased':              (36,  4,  7,  20, False),
    'eye_openness':            (54,  7,  8,  20, False),
    'eye_wrinkles':            (28,  6,  8,  20, False),
    'eyebrow_thickness':       (37,  8,  9,  20, False),
    'face_roundness':          (37,  0,  5,  20, False),
    'fearful_eyes':            (54,  4, 10,  20, False),
    'hairline':                (21,  4,  5, -20, False),
    'happy_frizzy_hair':       (30,  0,  8,  20, False),
    'happy_elderly_lady':      (27,  4,  7,  20, False),
    'head_angle_up':           (11,  1,  4,  20, False),
    'huge_grin':               (28,  4,  6,  20, False),
    'in_awe':                  (23,  3,  6, -15, False),
    'wide_smile':              (23,  3,  6,  20, False),
    'large_jaw':               (22,  3,  6,  20, False),
    'light_lr':                (15,  8,  9,  10, False),
    'lipstick_and_age':        (34,  6, 11,  20, False),
    'lipstick':                (34, 10, 11,  20, False),
    'mascara_vs_beard':        (41,  6,  9,  20, False),
    'nose_length':             (51,  4,  5, -20, False),
    'elderly_woman':           (34,  6,  7,  20, False),
    'overexposed':             (27,  8, 18,  15, False),
    'screaming':               (35,  3,  7, -15, False),
    'short_face':              (32,  2,  6, -20, False),
    'show_front_teeth':        (59,  4,  5,  40, False),
    'smile':                   (46,  4,  5, -20, False),
    'straight_bowl_cut':       (20,  4,  5, -20, False),
    'sunlight_in_face':        (10,  8,  9,  10, False),
    'trimmed_beard':           (58,  7,  9,  20, False),
    'white_hair':              (57,  7, 10, -24, False),
    'wrinkles':                (20,  6,  7, -18, False),
    'boyishness':              (8,   2,  5,  20, False),
}

def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes

# Setup required image transformations
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def load_hfgi_model(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['is_train'] = False
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    return net


"""
The function takes a preprocessed aligend and transcfomed image tensor and convert it into its latent vector
Using the pretrained HFGI model
Then returns an image along with its latent vector
"""
def img_2_latent_proj(transformed_img, net):
    with torch.no_grad():
        x = transformed_img.unsqueeze(0).cuda()

        # tic = time.time()
        latent_codes = get_latents(net, x)
        
        # calculate the distortion map
        imgs, _ = net.decoder([latent_codes[0].unsqueeze(0).cuda()],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        result_image, lat = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        # toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))
    
    return tensor2im(result_image[0]), latent_codes

def process_with_hfgi(img, net, predictor, detector):
    # #img = img.resize((256,256))
    aligned_img = align_face(img, predictor, detector)
    transformed_img = transform_img(aligned_img)
    projected_img, latent_vec = img_2_latent_proj(transformed_img, net)

    return aligned_img, transformed_img, projected_img, latent_vec


def project_change_inferfacegan(transformed_img, net, edit_direction, edit_degree, editor):
    with torch.no_grad():
        x = transformed_img.unsqueeze(0).cuda()

        # tic = time.time()
        latent_codes = get_latents(net, x)
        
        # calculate the distortion map
        imgs, _ = net.decoder([latent_codes[0].unsqueeze(0).cuda()],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        result_image, img_latent = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        # toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))

    img_edit, edit_latents = editor.apply_interfacegan(latent_codes[0].unsqueeze(0).cuda(), edit_direction, factor=edit_degree)
    # align the distortion map
    img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=(256,256) , mode='bilinear')
    res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

    # fusion
    conditions = net.residue(res_align)
    result, _ = net.decoder([edit_latents],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)

    result = torch.nn.functional.interpolate(result, size=(256,256) , mode='bilinear')
    

    return result, edit_latents, latent_codes

def manipulate_one_inter(transformed_img, net, dir_torch):

    edit_degree_max = 5
    edit_degree_min = -5
    num_steps = 60

    new_steps = num_steps-int(num_steps/2)

    step_size = (edit_degree_max-edit_degree_min)/new_steps
    edit_degree = 0

    interp_images = []
    for i in range(0,num_steps+1):
        # result, result_latent = project_change_inferfacegan(transformed_img, net, 2*dir_torch+age_direction, edit_degree)
        result, result_latent, org_latent  = project_change_inferfacegan(transformed_img, net, dir_torch, edit_degree)
        if (i>=0 and i<int(num_steps/4)):
            edit_degree = edit_degree+step_size
            interp_images.append(np.array(tensor2im(result[0])))
        elif (i>=int(num_steps/4) and i<int(num_steps*(3/4))):
            edit_degree = edit_degree-step_size
            interp_images.append(np.array(tensor2im(result[0])))
        else:
            edit_degree = edit_degree+step_size
            interp_images.append(np.array(tensor2im(result[0])))

    return interp_images


def make_random_video_inter(transformed_img, net, all_directions, tot_directions=10):
    
    all_interps = []
    for i in range(0,tot_directions):
        direction_path = random.choice(all_directions)
        dir_torch = load_direction(direction_path)
        # #edit_degree = 1.5
        print(direction_path.split('/')[-1])
        # print(dir_torch.max())

        if dir_torch.max()<0.1:
            scale_direction = 0.14/dir_torch.max()
            dir_torch = dir_torch*scale_direction

        # print(dir_torch.max())

        interp_images = manipulate_one_inter(transformed_img, net, dir_torch)
        all_interps.append(interp_images)

    return all_interps

def load_direction(direction_path):
    dir_numpy = np.load(direction_path)
    dir_torch = torch.from_numpy(dir_numpy).float().cuda()

    dir_torch = dir_torch[0].unsqueeze(0)
    return dir_torch



def project_change_ganspace(transformed_img, net, edit_direction, edit_degree, ganspace_pca, editor):
    with torch.no_grad():
        x = transformed_img.unsqueeze(0).cuda()

        # tic = time.time()
        latent_codes = get_latents(net, x)
        
        # calculate the distortion map
        imgs, _ = net.decoder([latent_codes[0].unsqueeze(0).cuda()],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        result_image, img_latent = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        # toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))

    
    edit_direction = (edit_direction[0],edit_direction[1],edit_direction[2],edit_degree)

    img_edit, edit_latents = editor.apply_ganspace(latent_codes[0].unsqueeze(0).cuda(), ganspace_pca, [edit_direction])
    # align the distortion map
    img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=(256,256) , mode='bilinear')
    res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))
    conditions = net.residue(res_align)
    result, _ = net.decoder([edit_latents],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
    result = torch.nn.functional.interpolate(result, size=(256,256) , mode='bilinear')

    return result, edit_latents, latent_codes

def manipulate_one_ganspace(transformed_img, net, edit_direction, ganspace_pca):
    
    edit_degree_max = 5*4
    edit_degree_min = -5*4
    num_steps = 60

    new_steps = num_steps-int(num_steps/2)

    step_size = (edit_degree_max-edit_degree_min)/new_steps
    edit_degree = 0

    interp_images = []
    for i in range(0,num_steps+1):
        # print(edit_degree)
        result, result_latent, org_latent = project_change_ganspace(transformed_img, net, edit_direction, edit_degree, ganspace_pca)

        if (i>=0 and i<int(num_steps/4)):
            edit_degree = edit_degree+step_size
            interp_images.append(np.array(tensor2im(result[0])))
        elif (i>=int(num_steps/4) and i<int(num_steps*(3/4))):
            edit_degree = edit_degree-step_size
            interp_images.append(np.array(tensor2im(result[0])))
        else:
            edit_degree = edit_degree+step_size
            interp_images.append(np.array(tensor2im(result[0])))

    return interp_images


def make_random_video_ganspace(transformed_img, net, ganspace_directions, ganspace_pca, tot_directions=3):
    all_interps = []
    all_directions = list(ganspace_directions.keys())
    
    for i in range(0,tot_directions):
        direction_path = random.choice(all_directions)
        print(direction_path)
        edit_direction = ganspace_directions[direction_path]

        # print(dir_torch.max())
        interp_images = manipulate_one_ganspace(transformed_img, net, edit_direction, ganspace_pca)
        all_interps.append(interp_images)

    return all_interps
