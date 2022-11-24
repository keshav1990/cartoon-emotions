
## ----------------------------------------------------------------------
## downloading Cartoon Models 
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
CODE_DIR = 'VToonify'


os.chdir(CODE_DIR)
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'checkpoint')
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'data')
OUT_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'output')

def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../checkpoint/. """
    #current_directory = os.getcwd()
    save_path = MODEL_DIR
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url

MODEL_PATHS = {
    "encoder": {"id": "1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej", "name": "encoder.pt"},
    "faceparsing": {"id": "1jY0mTjVB8njDh6e0LP_2UxuRK3MnjoIR", "name": "faceparsing.pth"},
    "arcane_exstyle": {"id": "1TC67wRJkdmNRZTqYMUEFkrhWRKKZW40c", "name": "exstyle_code.npy"},
    "caricature_exstyle": {"id": "1xr9sx_WmRYJ4qHGTtdVQCSxSo4HP3-ip", "name": "exstyle_code.npy"},
    "cartoon_exstyle": {"id": "1BuCeLk3ASZcoHlbfT28qNru4r5f-hErr", "name": "exstyle_code.npy"},
    "pixar_exstyle": {"id": "1yTaKuSrL7I0i0RYEEK5XD6GI-y5iNUbj", "name": "exstyle_code.npy"},
    "arcane000": {"id": "1pF4fJ8acmawMsjjXo4HXRIOXeZR8jLVh", "name": "generator.pt"},
    "arcane077": {"id": "16rLTF2oC0ZeurnM6hjrfrc8BxtW8P8Qf", "name": "generator.pt"},
    "caricature039": {"id": "1C1E4WEoDWzl0nAxR9okKffFmlMOENbeF", "name": "generator.pt"},
    "caricature068": {"id": "1B1ko1x8fX2aJ4BYCL12AnknVAi3qQc8W", "name": "generator.pt"},
    "cartoon026": {"id": "1YJYODh_vEyUrL0q02okjcicpJhdYY8An", "name": "generator.pt"},
    "cartoon299": {"id": "101qMUMfcI2qDxEbfCBt5mOg2aSqdTaIt", "name": "generator.pt"},
    "pixar052": {"id": "16j_l1x0DD0PjwO8YdplAk69sh3-v95rr", "name": "generator.pt"},
    "cartoon": {"id": "11s0hwhZWTLacMAzZH4OU-o3Qkp54h30J", "name": "generator.pt"},
}

# download pSp encoder and face parsinf network
path = MODEL_PATHS["encoder"]
download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
print(download_command)
os.system(download_command)
path = MODEL_PATHS["faceparsing"]
download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
print(download_command)
os.system(download_command)

style_type = 'cartoon026'
# download vtoonify
path = MODEL_PATHS[style_type]
download_command = get_download_model_command(file_id=path["id"], file_name = style_type + '_' + path["name"])
print(download_command)
os.system(download_command)
# download extrinsic style code
path = MODEL_PATHS[style_type[:-3]+'_exstyle']
download_command = get_download_model_command(file_id=path["id"], file_name = style_type[:-3] + '_' + path["name"])
print(download_command)
os.system(download_command)


# download the style code and the vtoonify-Dsd
path = MODEL_PATHS['cartoon_exstyle']
download_command = get_download_model_command(file_id=path["id"], file_name = 'cartoon_exstyle_code.npy')
print(download_command)
os.system(download_command)
path = MODEL_PATHS['cartoon']
download_command = get_download_model_command(file_id=path["id"], file_name = 'cartoon_generator.pt')
print(download_command)
os.system(download_command)
## ----------------------------------------------------------------------


## Download Superresolution Models
os.chdir('../GPEN/')
print(os.listdir())
os.system('wget "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116208&Signature=hBgvVvKVSNGeXqT8glG%2Bd2t2OKc%3D" -O weights/GPEN-BFR-512.pth')
os.system('wget "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116085&Signature=GlUNW6%2B8FxvxWmE9jKIZYOOciKQ%3D" -O weights/RetinaFace-R50.pth')
os.system('wget "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/realesrnet_x2.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1962694780&Signature=lI%2FolhA%2FyigiTRvoDIVbtMIyhjI%3D" -O weights/realesrnet_x2.pth')
os.system('wget "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116134&Signature=bnMwU1JogmNbARto6G%2B7iaJQCHs%3D" -O weights/ParseNet-latest.pth')
os.chdir('../')

## ----------------------------------------------------------------------
