img_size = (256,256)

import sys
sys.path.append('../face_inversion_utils/repositories_used/HFGI/')
sys.path.append('../face_inversion_utils/')

###from utils.face_alignment_utils import align_face
from image_alignment_utils import align_face_hfgi
from glob import glob
import os
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
    transforms.Resize(img_size),
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
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=img_size , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=img_size , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        result_image, lat = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        # toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))
    
    return tensor2im(result_image[0]), latent_codes

def process_with_hfgi(img, net, predictor, detector):
    aligned_img = align_face_hfgi(img, predictor, detector)
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
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=img_size , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=img_size , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        result_image, img_latent = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        # toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))

    img_edit, edit_latents = editor.apply_interfacegan(latent_codes[0].unsqueeze(0).cuda(), edit_direction, factor=edit_degree)
    # align the distortion map
    img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=img_size , mode='bilinear')
    res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

    # fusion
    conditions = net.residue(res_align)
    result, _ = net.decoder([edit_latents],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)

    #result = torch.nn.functional.interpolate(result, size=img_size , mode='bilinear')
    

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
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=img_size , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=img_size , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        result_image, img_latent = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        # toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))

    
    edit_direction = (edit_direction[0],edit_direction[1],edit_direction[2],edit_degree)

    img_edit, edit_latents = editor.apply_ganspace(latent_codes[0].unsqueeze(0).cuda(), ganspace_pca, [edit_direction])
    # align the distortion map
    img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=img_size , mode='bilinear')
    res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))
    conditions = net.residue(res_align)
    result, _ = net.decoder([edit_latents],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
    #result = torch.nn.functional.interpolate(result, size=img_size , mode='bilinear')

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

disp_size = (1024,1024)
figsize = (20,20)

def img_2_manipulated_interface(img, net, predictor,detector,dir_torch, edit_degree, editor):
    aligned_img, transformed_img, projected_img, latent_vec = process_with_hfgi(img, net, predictor, detector)
    result, result_latent, orig_latent = project_change_inferfacegan(transformed_img, net, dir_torch, edit_degree,editor)

    return tensor2im(result[0]), result_latent

def img_2_manipulated_ganspace(img, net, predictor, detector, edit_degree, editor,ganspace_pca,edit_direction):
    aligned_img, transformed_img, projected_img, latent_vec = process_with_hfgi(img, net, predictor, detector)
    result, result_latent, org_latent = project_change_ganspace(transformed_img, net, edit_direction, edit_degree, ganspace_pca, editor)

    return tensor2im(result[0]), result_latent
    
def cartoonize_image_hfgi(frame,style_id,num_imgs=10,padding=[200,200,200,200]):
    scale = 1
    kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
    # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
    # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
    paras = get_video_crop_parameter(frame, landmarkpredictor, padding=padding)
    # if paras is not None:
    #     h,w,top,bottom,left,right,scale = paras
    #     H, W = int(bottom-top), int(right-left)
    #     # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
    #     if scale <= 0.75:
    #         frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
    #     if scale <= 0.375:
    #         frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
    #     frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
    #     x = transform(frame).unsqueeze(dim=0).to(device)
    # else:
    #     print('no face detected!')

    x = transform(frame).unsqueeze(dim=0).to(device)
    with torch.no_grad():
        I = align_face(frame, landmarkpredictor)
        I = transform(I).unsqueeze(dim=0).to(device)
        s_w = pspencoder(I)


        s_w = vtoonify.zplus2wplus(s_w).repeat(len(style_id), 1, 1)
        s_w[:,:7] = exstyles[style_id,:7]
        x = x.repeat(len(style_id), 1, 1, 1)
        # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
        # followed by downsampling the parsing maps
        x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                            scale_factor=0.5, recompute_scale_factor=False).detach()
        # we give parsing maps lower weight (1/16)
        inputs = torch.cat((x, x_p/16.), dim=1)
        # d_s has no effect when backbone is toonify
        y_tilde = vtoonify(inputs, s_w, d_s = 0.6)        
        y_tilde = torch.clamp(y_tilde, -1, 1)


    # viz = torchvision.utils.make_grid(y_tilde, num_imgs, 2)
    # visualize(viz.cpu(), 50)

    results = []
    with torch.no_grad():
        for i in range(num_imgs):
            d_s = i / float(num_imgs)
            y_tilde = vtoonify(inputs, s_w, d_s = d_s)  
            y_tilde = torch.clamp(y_tilde, -1, 1)
            results += [y_tilde.cpu()]


    # vis = torchvision.utils.make_grid(torch.cat(results, dim=0), num_imgs, 2)

    return results

def load_all_directions(path_directions,device):
    all_direction_paths = glob(os.path.join(path_directions,'*.npy')) + glob(os.path.join(path_directions,'*.pt'))
    dir_dict = {}
    for i in range(0,len(all_direction_paths)):
        dir_path = all_direction_paths[i]
        dir_name = dir_path.split('/')[-1]
        dir_name,dir_ext = os.path.splitext(dir_name)
        if dir_ext=='.npy':
            dir_dict[dir_name] = load_direction(dir_path)
        if dir_ext=='.pt':
            dir_dict[dir_name] = torch.load(dir_path).to(device)
    
    
    return dir_dict


## disgust
def make_annoy(img_org,net,predictor, detector, all_directions_dict,editor):
    manipulated_img,manipulated_latent = img_2_manipulated_interface(img_org,net,predictor, detector, 
                                                                     all_directions_dict['stg2']['gender'], -1, editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, 
                                                                     all_directions_dict['inter']['age'],-1, editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, 
                                                                     all_directions_dict['generators']['emotion_disgust'], 8, editor)

    return manipulated_img

def make_annoy(img_org,net,predictor, detector, all_directions_dict,editor,intensity_annoy):
    manipulated_img,manipulated_latent = img_2_manipulated_interface(img_org,net,predictor, detector, 
                                                                     all_directions_dict['stg2']['gender'], intensity_annoy['gender'], editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, 
                                                                     all_directions_dict['generators']['eyes_open'],intensity_annoy['eyes_open'], editor)
    
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, 
                                                                     all_directions_dict['inter']['age'],intensity_annoy['age'], editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, 
                                                                     all_directions_dict['generators']['emotion_disgust'], intensity_annoy['emotion_disgust'], editor)

    return manipulated_img

## happy
def make_smile(img_org,net,predictor, detector, all_directions_dict,editor,intensity_smile):
    manipulated_img,manipulated_latent = img_2_manipulated_interface(img_org,net,predictor, detector, all_directions_dict['inter']['smile'], intensity_smile['smile'], editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['stg2']['gender'], intensity_smile['gender'] , editor)

    return manipulated_img

## sad
def make_sad(img_org,net,predictor, detector, all_directions_dict,editor):
    manipulated_img,manipulated_latent = img_2_manipulated_interface(img_org,net,predictor, detector, all_directions_dict['generators']['emotion_easy'], 5, editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['generators']['emotion_sad'], 13, editor)
    # manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['stg2']['gender'], -1, editor)
    # manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, direction_artifact, -3, editor)
    return manipulated_img

def make_sad(img_org,net,predictor, detector, all_directions_dict,editor,intensity_sad):
    manipulated_img,manipulated_latent = img_2_manipulated_interface(img_org,net,predictor, detector, all_directions_dict['generators']['emotion_easy'], intensity_sad['emotion_easy'], editor)
    # manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['generators']['eyes_open'],-3, editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['generators']['emotion_sad'], intensity_sad['emotion_sad'], editor)
    # manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['stg2']['gender'], -1, editor)
    # manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, direction_artifact, -7, editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['generators']['eyes_open'],intensity_sad['eyes_open'], editor)
    
    return manipulated_img

## angry
def make_angry(img_org,net,predictor, detector, all_directions_dict,editor,intensity_angry):
    manipulated_img,manipulated_latent = img_2_manipulated_interface(img_org,net,predictor, detector, all_directions_dict['generators']['emotion_easy'], intensity_angry['emotion_easy'], editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['generators']['emotion_happy'], intensity_angry['emotion_happy'], editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['generators']['emotion_angry'], intensity_angry['emotion_angry'], editor)
    manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, all_directions_dict['inter']['age'], intensity_angry['age'], editor)
    # manipulated_img,manipulated_latent = img_2_manipulated_interface(manipulated_img,net,predictor, detector, direction_artifact, -7, editor)
    return manipulated_img
