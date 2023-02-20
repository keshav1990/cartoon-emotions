import cv2
import PIL
import os
import json

def load_img(img_path, rgb=True, size=False):
    img = cv2.imread(img_path)
    if rgb==True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size)
    
    return img

def load_img_pil(img_path, resize_dims=None):
    original_image = PIL.Image.open(img_path)
    original_image = original_image.convert("RGB")
    if resize_dims !=None:
        original_image = original_image.resize(resize_dims)

    return original_image


def read_matched_images(matched_img_paths):
    matched_imgs = []
    for i in range(0,len(matched_img_paths)):
        matched_imgs.append(load_img_pil(matched_img_paths[i]))

    return matched_imgs

def save_json(img_name,json_data,path_output):
    ## sp_name,image_ext = os.path.splitext(img_name)
    
    with open(os.path.join(path_output, img_name+'.json'), 'w') as outfile:
        json.dump(json_data, outfile)
