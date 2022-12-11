import os
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.geometry import Polygon
import glob


def mask2poly(mask):
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.uint8), mask=(mask > 0),
                                        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shapely.geometry.shape(shape))
    return all_polygons


def poly2mask(polygons, shape):
    binary_mask = rasterio.features.rasterize(
        polygons,
        out_shape=shape
    )
    return (binary_mask * 255).astype(np.uint8)


class MultiLabelScantronic(Dataset):
    def __init__(self, images_dir, masks_dir, pickle_data, img_size, suffix, data_ext='.npy'):
        self.img_size = img_size
        split_stems = [Path(f).stem for f in pickle_data]
        images_orig = glob.glob(f'{images_dir}/*{data_ext}')
        masks_orig = glob.glob(f'{masks_dir}/*{suffix}.png')
        intersected_names = list(
            set(
                str(Path(f).stem) for f in images_orig
            ).intersection(
                str(Path(f).stem).strip('_machine_mask') for f in masks_orig
            )
        )
        final_intersection_names = set(intersected_names).intersection(split_stems)
        self.image_paths = [
            f for f in glob.glob(f'{images_dir}/*{data_ext}')
            if str(Path(f).stem) in final_intersection_names
        ]
        self.image_paths.sort()

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def get_image(item):
        suffix = Path(item).suffix
        if suffix == '.npy':
            image = np.load(item)
            if image.max() > 255:
                image = (image / image.max()) * 255.
            image = Image.fromarray(image.astype(np.uint8))
        elif suffix in ['.png', '.jpg', '.jpeg']:
            image = Image.open(item)
        else:
            raise NotImplemented
        return image

    def __getitem__(self, item):
        image = self.get_image(self.image_paths[item])

        mask_path = self.image_paths[item].replace('train_target_files', 'annotations').replace('.png', '_machine_mask.png')
        mask = Image.open(mask_path)

        # resize image and mask
        image_copy = image.resize(self.img_size, resample=Image.BICUBIC)
        mask_copy = mask.resize(self.img_size, resample=Image.NEAREST)

        # preprocess image and mask
        img = self.preprocess(image_copy, scale=1., is_mask=False)
        mask = self.preprocess(mask_copy, scale=1., is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

    def __len__(self):
        return len(self.image_paths)


def main():
    data_folder = '../../converter/model_forge/f2e4a3a6-f9d7-49fc-a9da-79fb325c3899'
    images_dir = os.path.join(data_folder, 'train_target_files')
    masks_dir = os.path.join(data_folder, 'annotations')
    dataloader = MultiLabelScantronic(images_dir, masks_dir, img_size=[1280, 1280])
    result = dataloader.__getitem__(0)


if __name__ == '__main__':
    main()
