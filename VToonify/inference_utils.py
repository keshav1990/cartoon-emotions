
import sys
sys.path.append(".")
sys.path.append("..")

import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
from image_enhancement_gpen import gpen_enhance,faceenhancer

import torch
from torchvision import transforms

import dlib
import torch.nn.functional as F
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import load_psp_standalone, get_video_crop_parameter, tensor2cv2


## load_model
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu' 

vtoonify = VToonify(backbone = 'dualstylegan')
vtoonify.load_state_dict(torch.load(os.path.join('checkpoint/', 'cartoon_generator.pt'), map_location=lambda storage, loc: storage)['g_ema'])
vtoonify.to(device)

exstyles = np.load(os.path.join('checkpoint/', 'cartoon_exstyle_code.npy'), allow_pickle='TRUE').item()  
styles = []
with torch.no_grad(): 
    for stylename in exstyles.keys():
        exstyle = torch.tensor(exstyles[stylename]).to(device)
        exstyle = vtoonify.zplus2wplus(exstyle)
        styles += [exstyle]
exstyles = torch.cat(styles, dim=0)


parsingpredictor = BiSeNet(n_classes=19)
parsingpredictor.load_state_dict(torch.load(os.path.join('checkpoint', 'faceparsing.pth'), map_location=lambda storage, loc: storage))
parsingpredictor.to(device).eval()

pspencoder = load_psp_standalone(os.path.join('checkpoint', 'encoder.pt'), device)  

landmarkpredictor = dlib.shape_predictor('checkpoint/shape_predictor_68_face_landmarks.dat')
print('Models successfully loaded!')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])


def cartoonize_image(frame,style_id,num_imgs=10):
    scale = 1
    kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
    # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
    # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
    paras = get_video_crop_parameter(frame, landmarkpredictor, padding=[200,200,200,200])
    if paras is not None:
        h,w,top,bottom,left,right,scale = paras
        H, W = int(bottom-top), int(right-left)
        # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
        if scale <= 0.75:
            frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
        if scale <= 0.375:
            frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
        frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
        x = transform(frame).unsqueeze(dim=0).to(device)
    else:
        print('no face detected!')

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

def generate_cartoons(image_path, save_dir, style_id_list=[26,64,299], num_imgs=10, fps=10):


    for style_id in style_id_list:
        print('Generating results for style id:',style_id)
        style_id = [style_id]
        ## cartoonization requires RGB image
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = cartoonize_image(frame, style_id, num_imgs)

        
        img_name = os.path.split(image_path)[-1]

        img_style_name = os.path.splitext(img_name)[0]+'_style_'+str(style_id[0])
        frames_path = os.path.join(save_dir,img_style_name)


        if not os.path.isdir(save_dir):
            os.mkdir(save_dir) 
        if not os.path.isdir(frames_path):
            os.mkdir(frames_path)


        H,W = np.shape(results[0].squeeze())[1:]

        vid_name = img_style_name+'.mp4'
        vid_path = os.path.join(save_dir,vid_name)
        
        vid_writer = cv2.VideoWriter(vid_path, 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, (H,W))
        
        print('Enhancing and saving results for style id:',style_id)
        for i in tqdm(range(0,len(results))):
            img = results[i].squeeze()
            img = ((img.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img = gpen_enhance(img,faceenhancer) ## BGR images
            img = cv2.resize(img,(H,W))

            img_path = os.path.join(frames_path,str(i)+'.jpg')
            cv2.imwrite(img_path,img)
            vid_writer.write(img)

        vid_writer.release()
