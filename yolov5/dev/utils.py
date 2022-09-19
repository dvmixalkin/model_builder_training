import numpy as np
import torch
import cv2
from pathlib import Path
import json
import yaml
import os
from PIL import Image


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def get_image(path=None, k=1.):
    if path is None:
        raise ValueError
    img = cv2.imread(path)
    if k != 1.:
        img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)))
    # img = np.transpose(img, (2, 0, 1))
    return img


def read_file(path=None):
    suffix = Path(path).suffix
    with open(path, 'r') as stream:
        if suffix == '.json':
            data = json.load(stream)
        if suffix == '.yaml':
            data = yaml.safe_load(stream)
        if suffix == '.txt':
            data = stream.readlines()
    return data


def get_labels(lbl_path, style='yolo', img_size=None):
    data = read_file(path=lbl_path)

    if style == 'yolo':
        data = [[float(element) for element in line.strip().split()] for line in data]
        data = np.asarray(data)
        return data

    if style == 'absolute':
        labels = data[:, 0]
        boxes = data[:, 1:]
        return boxes, labels


def get_data(img_path, lbl_path, anno_path):
    image = get_image(img_path)
    annotations = get_labels(lbl_path)

    c, h, w = image.shape
    boxes = np.copy(annotations[:, 1:])
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h

    labels = annotations[:, 0]
    general_data = read_file(anno_path)
    class_names = general_data['names']


def draw_bboxes(image, bboxes, labels, scores=None, color=None):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy().astype(np.int16)
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().astype(np.int16)

    image = np.ascontiguousarray(image)
    if color is None:
        color = (0, 204, 0)
    if scores is None:
        for bbox, label in zip(bboxes[:, -4:], labels):
            cv2.rectangle(image, bbox[0:2].astype(int), bbox[2:4].astype(int), color, 2)
            cv2.putText(image, f'{label}', (bbox[0].astype(int), bbox[1].astype(int) + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    else:
        for bbox, label, score in zip(bboxes[:, -4:], labels, scores):
            cv2.rectangle(image, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(image, f'{label}-{score}', (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)

    return image


def load_npz(path_to_file, norm_values, channels, save_image, path_to_current_dataset_images):
    """
    Конвертация npz-массива в формат диапазон значений RGB(0-255)
    Args:
        path_to_file(str): путь до изображения
        norm_values(dict): нормировочные параметры для npz-матрицы
        channels(list): используемые каналы для дальшейшего препроцессинга
        save_image(bool): True или False флаг, сигнализирует о необходимости сохранения изображения
                            по пути path_to_current_dataset_images
        path_to_current_dataset_images(str):

    Returns:
        im: numpy-массив изображения
        path_to_save_image: путь к изображению
    """
    im = np.load(path_to_file)
    futher_array = []

    # ['raw_image_low', 'raw_image_high', 'clusterization_mask', 'z_number_mask', 'thickness_mask', 'weight_mask']:
    for key in im.keys():
        channel_max_value = im[key].max()
        target_values = norm_values[key]
        if isinstance(target_values, list):
            t_values = target_values[0] if channel_max_value <= target_values[0] else target_values[1]
            normalized_array = im[key] / t_values
        else:
            normalized_array = im[key] / target_values
        if normalized_array.max() > 1:
            normalized_array /= normalized_array.max()

        futher_array.append(normalized_array * 255.)

    im = np.stack(futher_array).transpose(1, 2, 0).astype(np.uint8)[..., channels]
    path_to_save_image = os.path.join(path_to_current_dataset_images, f'{Path(path_to_file).stem}.png')

    if save_image and not os.path.exists(path_to_save_image):
        if im.shape[2] > 3:
            np.save(path_to_save_image, im)
        else:
            Image.fromarray(im).save(path_to_save_image)

    return im, path_to_save_image
