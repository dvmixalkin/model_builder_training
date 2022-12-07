import argparse
import glob
import json
import numpy as np
import os
import pickle
import torch
import yaml

from pathlib import Path

import rasterio
import shapely
from rasterio import features
from shapely.geometry import Polygon
from PIL import Image


def mask2poly(mask):
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.uint8), mask=(mask > 0),
                                        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shapely.geometry.shape(shape))
    return all_polygons


def poly2mask(polygons, shape):
    if isinstance(polygons, shapely.geometry.Polygon):
        polygons = [polygons]
    binary_mask = rasterio.features.rasterize(
        polygons,
        out_shape=shape
    )
    return binary_mask.astype(np.uint8)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (int(x[:, 0]) + int(x[:, 2])) / 2  # x center
    y[:, 1] = (int(x[:, 1]) + int(x[:, 3])) / 2  # y center
    y[:, 2] = int(x[:, 2]) - int(x[:, 0])  # width
    y[:, 3] = int(x[:, 3]) - int(x[:, 1])  # height
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def parse_annotations(path_to_json, width=None, heigh=None):
    try:
        with open(path_to_json, 'r') as stream:
            data = json.load(stream)
    except:
        path_to_json = path_to_json.replace('.json', '_predict.json')
        with open(path_to_json, 'r', encoding='UTF-8') as stream:
            data = json.load(stream)

    boxes = []
    polygons = []
    labels = []
    for ann in data:
        label = ann['tnved'] if 'tnved' in ann and ann['tnved'] is not None else ann['id_category']
        points = np.ascontiguousarray(ann.get('points')) if 'points' in ann else None
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        labels.append(label)
        x = np.asarray([[x_min, y_min, x_max, y_max]], dtype=float)
        if width is not None and heigh is not None:
            rectangle = xyxy2xywhn(x, w=width, h=heigh)
        else:
            rectangle = xyxy2xywh(x)
        boxes.append(rectangle.tolist()[0])
        polygons.append(points)
    return boxes, polygons, labels


class Pickler:
    def __init__(
            self, root_folder: str,
            cfg_filename: str,
            general_classes: bool = True,
            train_fraction: float = .9,
            pickle_filename: str = 'train_dataset.pkl'
    ):
        """
        general_classes - использовать разметку из словаря сопоставления категорий с классами.
                          словарь брать из файла с конфигами(по ключу "classes_map")
        """
        self.root_folder = root_folder
        if not os.path.exists(root_folder):
            raise FileNotFoundError
        self.cfg_filename = cfg_filename
        self.general_classes = general_classes
        self.train_fraction = train_fraction
        self.pickle_filename = pickle_filename

    def process(self, train_name: str, data_ext: str = '.npy', anno_ext: str = '.json'):
        if not data_ext.startswith('.'):
            data_ext = '.' + data_ext
        if not anno_ext.startswith('.'):
            anno_ext = '.' + anno_ext

        train_dir = os.path.join(self.root_folder, train_name)
        if not os.path.exists(train_dir):
            return None, 0  # raise FileNotFoundError
        else:
            npy_folder = os.path.join(train_dir, 'train_target_files')
            if not os.path.exists(npy_folder):
                return None, 0  # raise FileNotFoundError
            else:
                npy_dict, npy_files = {}, []
                # for file in os.listdir(npy_folder):
                for file in glob.glob(npy_folder + f'/*{data_ext}'):
                    stem = str(Path(file).stem)
                    name = stem.split('_height_')[0]
                    npy_files.append(name)
                    npy_dict[name] = {
                        'original_name': stem,
                        'size': [int(value) for value in stem.split('_height_')[1].split('_width_')]
                    }

            annotation = os.path.join(train_dir, 'annotations')
            if not os.path.exists(annotation):
                return None, 0  # raise FileNotFoundError
            else:
                json_files = [str(Path(file).stem).strip('_predict') for file in os.listdir(annotation)]

            config_path = os.path.join(self.root_folder, train_name, self.cfg_filename)
            if not os.path.exists(config_path):
                return None, 0
            with open(config_path, 'r') as yaml_stream:
                classes_map = yaml.safe_load(yaml_stream)['classes_map']

        intersected_names = list(set(npy_files).intersection(set(json_files)))
        # if set(npy_files) == set(json_files):
        if len(intersected_names) > 0:
            class_mapper = {}
            cumulative_dict = {'train': {}, 'val': {}, 'configs': None}
            for filename in intersected_names:
                npy_file = os.path.join(npy_folder, npy_dict[filename]['original_name'] + data_ext)
                height, width = npy_dict[filename]['size']
                json_file = os.path.join(annotation, filename + anno_ext)

                boxes, polygons, labels = parse_annotations(json_file, width=width, heigh=height)
                anno = []
                mask = np.zeros([height, width])
                for lbl, box, polygon in zip(labels, boxes, polygons):
                    cls_name = lbl
                    if self.general_classes:
                        cls_name = classes_map[lbl]
                    if cls_name not in class_mapper:
                        class_mapper[cls_name] = len(class_mapper)
                    cls_id = class_mapper[cls_name]
                    polygon = Polygon(polygon)
                    mask += poly2mask(polygon, [height, width]) * (cls_id + 1)
                    anno.append([cls_id, *box])

                seed = np.random.rand()
                split = 'train' if seed < self.train_fraction else 'val'
                cumulative_dict[split][npy_file] = anno
                save_name = npy_file\
                    .replace('train_target_files', 'annotations')\
                    .replace(data_ext, '_machine_mask.png')
                Image.fromarray(mask).convert("L").save(save_name)

            cumulative_dict['configs'] = {
                'machine_mapper': class_mapper,
                'human_mapper': classes_map,
            }

            if cumulative_dict == {
                'train': {}, 'val': {},
                'configs': {
                    'machine_mapper': {},
                    'human_mapper': classes_map
                }
            }:
                return None, 0

            pickle_path = os.path.join(train_dir, self.pickle_filename)
            with open(pickle_path, 'wb') as stream_:
                pickle.dump(obj=cumulative_dict, file=stream_)

            return b'', 1
        else:
            return None, 0


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', type=str, default='model_forge', help='path to root_folder')
    parser.add_argument('--cfg-filename', type=str, default='train_settings.yaml', help='path to config file')
    parser.add_argument('--train-name', type=str, default='f2e4a3a6-f9d7-49fc-a9da-79fb325c3899',
                        help='path to train folder')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    pickler = Pickler(opt.root_folder, opt.cfg_filename)
    return pickler.process(opt.train_name, data_ext='.png')


if __name__ == '__main__':
    main()
