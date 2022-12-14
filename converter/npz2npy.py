import io
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def get_data(
        root_folder='model_forge',
        train_name='f2e4a3a6-f9d7-49fc-a9da-79fb325c3899',
        image_folder='train_target_files',
        filename=''  # fbdab1b0-4d19-4a91-89c2-be21cf6ba5a2.npz
):
    test_path = os.path.join(root_folder, train_name, image_folder, filename)
    with open(test_path, 'rb') as stream:
        npz_bytes = stream.read()
    save_path = os.path.join(str(Path(test_path).parent), str(Path(test_path).stem))
    return npz_bytes, save_path


def process_image(object_, channels):
    try:
        data = np.load(io.BytesIO(object_))
    except:
        raise NotImplementedError
    data_array = []
    try:
        for channel in channels:
            channel_data = data.get(channel)
            channel_data = (channel_data / channel_data.max()) * 255
            data_array.append(channel_data)
    except:
        raise NotImplementedError

    array_to_save = np.array(data_array, dtype=float)
    if array_to_save.shape[0] == 1:
        array_to_save = array_to_save[0]
    else:
        array_to_save = array_to_save.transpose(1, 2, 0)
    return array_to_save


def resize_array(array_to_save, divider=2.):
    resize_width = int(array_to_save.shape[1] / divider)
    resize_height = int(array_to_save.shape[0] / divider)
    dim = (resize_width, resize_height)
    return cv2.resize(array_to_save, dim, interpolation=cv2.INTER_AREA)


class Converter:
    def __init__(self, root_folder: str):
        self.root_folder = root_folder

    @staticmethod
    def process(npz_bytes: bytes, save_path: str = '',
                scale_factor: float = 1, channels: list = ['raw_image_low'],
                ext: str = '.npy'):
        if save_path == '':
            return None, 0
        if not isinstance(npz_bytes, bytes):
            return None, 0

        npy_array = process_image(npz_bytes, channels)
        h, w = npy_array.shape
        try:
            if ext == '.npy':
                np.save(file=save_path + f'_height_{h}_width_{w}' + '.npy', arr=resize_array(npy_array, divider=scale_factor))
            elif ext == '.png':
                path2save = save_path + f'_height_{h}_width_{w}' + ext
                Image.fromarray(resize_array(npy_array, divider=scale_factor)).convert('L').save(path2save)
            else:
                return None, 0
        except:
            return None, 0
        return b'', 1


def parse_opt():
    parser = argparse.ArgumentParser()
    npz_bytes, save_path = get_data(
        root_folder='model_forge',
        train_name='f2e4a3a6-f9d7-49fc-a9da-79fb325c3899',
        image_folder='train_target_files',
        filename='fbdab1b0-4d19-4a91-89c2-be21cf6ba5a2.npz')

    parser.add_argument('--root_folder', type=str, default='model_forge', help='root folder')
    parser.add_argument('--bytes', type=bytes, default=npz_bytes, help='npz bytes')
    parser.add_argument('--save_path', type=str, default=save_path, help='save path')
    parser.add_argument('--channels', type=list, default=['raw_image_low'], help='channels to process')
    parser.add_argument('--scale_factor', type=float, default=2., help='coeff to divide img')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    converter = Converter(root_folder=opt.root_folder)
    return converter.process(
        npz_bytes=opt.bytes,
        save_path=opt.save_path,
        scale_factor=opt.scale_factor,
        channels=opt.channels
    )


def main_loop():
    root_folder = 'model_forge'
    train_name = 'f2e4a3a6-f9d7-49fc-a9da-79fb325c3899'
    image_folder = 'train_target_files'
    im_fldr_path = os.path.join(root_folder, train_name, image_folder)

    converter = Converter(root_folder=root_folder)

    for im in tqdm([file for file in os.listdir(im_fldr_path) if '.npz' in file]):
        npz_bytes, save_path = get_data(
            root_folder='model_forge',
            train_name='f2e4a3a6-f9d7-49fc-a9da-79fb325c3899',
            image_folder='train_target_files',
            filename=im)

        converter.process(
            npz_bytes=npz_bytes,
            save_path=save_path,
            scale_factor=1,
            channels=['raw_image_low'],
            ext='.png'
        )


if __name__ == '__main__':
    # main()
    main_loop()
