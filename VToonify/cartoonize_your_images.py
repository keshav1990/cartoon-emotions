import argparse
from inference_utils import generate_cartoons
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path_data", help="path of input image or directory")
parser.add_argument("--save_dir", help="path to save resultant images", default='/content/cartoonization_results/')
parser.add_argument("--num_imgs", help="number of intermediate images you want to generate",type=int,default=10)
parser.add_argument("--fps", help="frames per second of the video",type=int,default=10)
parser.add_argument("--styles",help="cartoon styles you want to generate"
                    ,default=[26,64,299], action='store',nargs='*',dest='styles',type=int)

args = parser.parse_args()
print(args.path_data)


if os.path.isfile(args.path_data):
    image_path = args.path_data
    try:
        generate_cartoons(image_path, args.save_dir, style_id_list=args.styles, num_imgs=args.num_imgs, fps=args.fps)
    else:
        print('Error:',image_path)

elif os.path.isdir(args.path_data):
    all_paths = glob(os.path.join(args.path_data,'*'))
    for i in range(0,len(all_paths)):
        image_path = all_paths[i]
        try:
            generate_cartoons(image_path, args.save_dir, style_id_list=args.styles, num_imgs=args.num_imgs, fps=args.fps)
        else:
            print('Error:',image_path)
