import argparse
from glob import glob
import os

import sys
sys.path.append(".")
sys.path.append("..")

import os
import sys
sys.path.append('../face_inversion_utils/repositories_used/HFGI/')

import PIL
from PIL import Image
from glob import glob
import torch
from editings import latent_editor

import numpy as np
import cv2
from tqdm import tqdm
from image_enhancement_gpen import gpen_enhance,faceenhancer

from torchvision import transforms

import dlib
import torch.nn.functional as F
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import load_psp_standalone, get_video_crop_parameter, tensor2cv2

from hfgi_utils import load_hfgi_model,load_all_directions
import joblib 
import pickle
from hfgi_utils import make_sad,make_smile,make_angry,make_annoy


## load_model
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu' 

###########################################################################
def cartoonize_image_hfgi(frame,style_id,num_imgs=10,padding=[200,200,200,200]):
    scale = 1
    kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
    # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
    # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
    paras = get_video_crop_parameter(frame, landmarkpredictor, padding=padding)

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


def generate_cartoons_interface(image_path, save_dir, hfgi_net,predictor, detector, all_directions_dict, emotion,editor, 
                                 style_id_list=[26,64,299], num_imgs=10, fps=10,padding=[200,200,200,200]):

    
    for style_id in style_id_list:
        print('Generating results for style id:',style_id)
        style_id = [style_id]
        ##############################################
        ## cartoonization requires RGB image
        frame = cv2.imread(image_path)
        
        ## make 3 channel input
        if np.shape(frame)[-1]!=3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ###############################################

        # frame,laten_vect = img_2_manipulated_interface(PIL.Image.fromarray(frame),hfgi_net,predictor, detector, edit_direction, edit_degree, editor)

        if emotion=='happy':
            frame = make_smile(PIL.Image.fromarray(frame),net,predictor, detector, all_directions_dict,editor)
        if emotion=='sad':
            frame = make_sad(PIL.Image.fromarray(frame),net,predictor, detector, all_directions_dict,editor)
        if emotion=='angry':
            frame = make_angry(PIL.Image.fromarray(frame),net,predictor, detector, all_directions_dict,editor)
        if emotion=='annoy':
            frame = make_annoy(PIL.Image.fromarray(frame),net,predictor, detector, all_directions_dict,editor)
        
        
        delta_imgs = 2 # to smoothen the initial images
        frame = cv2.resize(np.array(frame),(400,400))
        results = cartoonize_image_hfgi(frame, style_id, num_imgs+delta_imgs,padding=padding)

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
        for i in tqdm(range(delta_imgs,len(results))):
            img = results[i].squeeze()
            img = ((img.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img = gpen_enhance(img,faceenhancer) ## BGR images
            img = cv2.resize(img,(H,W))

            img_path = os.path.join(frames_path,str(i-delta_imgs)+'.jpg')
            cv2.imwrite(img_path,img)
            vid_writer.write(img)

        vid_writer.release()
#########################################################################

## load the toonify models

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


## load the manipulation models

predictor = landmarkpredictor#dlib.shape_predictor("../face_inversion_utils/pretrained_models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
net = load_hfgi_model('../face_inversion_utils/pretrained_models/ckpt.pt')

# ## interfaceGAN
editor = latent_editor.LatentEditor(net.decoder)


## load the directions
clf = joblib.load('../face_inversion_utils/pretrained_models/artifacts_model/artifacts_model.joblib')
attribute_direction = clf.coef_.reshape((18, 512))
direction_artifact = torch.from_numpy(attribute_direction[0]).float().cuda()


directions_generators = load_all_directions('../face_inversion_utils/pretrained_models/generators_with_stylegan2_directions/',device)
directions_stg2 = load_all_directions('../face_inversion_utils/pretrained_models/stylegan2directions/',device)
directions_inter = load_all_directions('../face_inversion_utils/repositories_used/HFGI/editings/interfacegan_directions/',device)

all_directions_dict = {}
all_directions_dict['generators'] = directions_generators
all_directions_dict['stg2'] = directions_stg2
all_directions_dict['inter'] = directions_inter
##################################################################################################



parser = argparse.ArgumentParser()
parser.add_argument("--path_data", help="path of input image or directory",type=str)
parser.add_argument("--save_dir", help="path to save resultant images", default='/content/cartoonization_results/')
parser.add_argument("--num_imgs", help="number of intermediate images you want to generate",type=int,default=10)
parser.add_argument("--fps", help="frames per second of the video",type=int,default=10)
parser.add_argument("--styles",help="cartoon styles you want to generate"
                    ,default=[26,64,299], action='store',nargs='*',dest='styles',type=int)
parser.add_argument("--emotion",help="select the emotion from, happy,sad,angry,annoy",type=str,default="happy")

##args = parser.parse_args(args=[])
args = parser.parse_args()

assert args.emotion=="happy" or args.emotion=="sad" or args.emotion=="angry" or args.emotion=="annoy"
print(args.path_data)


emotion = args.emotion
all_test_images = glob(os.path.join(args.path_data,'*'))
save_dir = args.save_dir 
style_id_list = args.styles

#####################################################################

print('Emotion:',emotion)
for image_path in all_test_images:
    
    try:
        generate_cartoons_interface(image_path, save_dir, net,predictor, detector, all_directions_dict, emotion, editor, 
                                        style_id_list=style_id_list, num_imgs=args.num_imgs, fps=args.fps,padding=[2,2,2,2])
    
    except:
        print('------------------------')
        print("Error with image:",image_path.split('/')[-1])
