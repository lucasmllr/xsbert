import zipfile
import os
from os.path import join, exists
from os import PathLike
import wget
import sys

from . import XSMPNet, XSRoberta


def load_checkpoint(model_name: str):

    path = join('checkpoints', model_name)
    if not exists(path):
        zip_path = f'checkpoints/{model_name}.zip'
        if not zipfile.is_zipfile(zip_path):
            print('fetching checkpoint from LFS')
            os.system(f'git lfs pull --exclude="" --include="{zip_path}"')
        print('unzipping')
        with zipfile.ZipFile(zip_path, 'r') as f:
            f.extractall('checkpoints/')

    print('initializing')
    if model_name.endswith('roberta'):
        model = XSRoberta(path)
    elif model_name.endswith('mpnet'):
        model = XSMPNet(path)

    return model


def progress(current, total, width):
  progress_message = f"downloading model: {int(current / total * 100)}%"
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

def load_model(name: str, model_dir: PathLike = '../xs_models/'):
    assert name in ['mpnet_cos', 'distilroberta_dot'], \
        'available models are: xs_mpnet and xs_distilroberta'
    if not exists(model_dir):
        os.makedirs(model_dir)
    path = join(model_dir, name)
    if not exists(path):
        zip_path = path + '.zip'
        if not exists(zip_path):
            url = f'https://www2.ims.uni-stuttgart.de/data/xsbert/{name}.zip'
            wget.download(url, zip_path, bar=progress)
            print()
        print('unzipping')
        with zipfile.ZipFile(zip_path, 'r') as f:
            f.extractall(model_dir)
    print('initializing')
    if name.startswith('roberta'):
        model = XSRoberta(path)
    elif name.startswith('mpnet'):
        model = XSMPNet(path)
    return model