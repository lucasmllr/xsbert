import zipfile
import os
from os.path import join, exists

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