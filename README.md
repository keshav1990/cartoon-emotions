# cartoonize_yourself

This repository contains the code for face cartoonization.

# Pipeline
Clone the repository

```
git clone https://github.com/waleedrazakhan92/cartoonize_yourself.git
cd cartoonize_yourself/
```
Then download all the necessary models using the *download_models.py* file.
```
python3 download_models.py
```
You can also download the models from the reference repositories:
1) https://github.com/williamyang1991/VToonify
2) https://github.com/yangxy/GPEN

If you're manually downloading the models then download all the models from VToonify into the 
```VToonify/checkpoint/``` folder and the models from GPEN are to be downloaded in the ```GPEN/weights``` folder.

The next step is to install Ninja using the following command:
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```

Now change the directory to VToonify and you're all set:
```cd VToonify/```

Cartoonize yourself using the following command:
```
python3 cartoonize_your_images.py --path_data 'Path/to/input/image or folder of images' \
--styles 26 64 299\
--fps 10 \
--num_imgs 10 
```

## Arguments explained:
* --path_data           This can be a path of an input image or a folder of images
* --save_dir            This is the path of the directory where results will be saved
* --styles              Choose any style or multiple styles from the available file [8, 26, 64, 153, 299]
* --fps                 Frames per second of the result video
* --num_imgs            Number of intermediate images of the particular style
