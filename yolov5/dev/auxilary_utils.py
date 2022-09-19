import glob
import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import yaml
from PIL import Image


def te_wrapper(dictionary, key):
    try:
        return dictionary.get(key)
    except:
        raise ValueError(f'Dictionary does not contain {key} key.')


def get_boxes(objects, return_polygons=False):

    objects_stack = []
    for part in objects:
        items = objects.get(part)
        if isinstance(items, list):
            tmp = [i for i in items if i.__contains__('tnved') and i.get('tnved') is not None]
            if tmp:
                objects_stack.extend(tmp)
        else:  # isinstance(items, dict):
            if items.__contains__('tnved'):
                objects_stack.append(items)

    rectangles = []
    if return_polygons:
        polygons = []
    for item in objects_stack:
        tnved = item.get('tnved')
        coordinates = item.get('points')
        np_coordinates = np.asarray(coordinates)
        box = [
            tnved,
            np_coordinates[:, 0].min(), np_coordinates[:, 1].min(),
            np_coordinates[:, 0].max(), np_coordinates[:, 1].max()
        ]
        rectangles.append(box)
        if return_polygons:
            polygons.append(coordinates)
    if return_polygons:
        return np.asarray(rectangles), polygons
    return np.asarray(rectangles)


def read_json(path):
    try:
        with open(path, encoding='utf-8', mode='r') as stream:
            json_file = json.load(stream)
    except:
        with open(path, encoding='windows-1252', mode='r') as stream:  # utf-8
            json_file = json.load(stream)
    return json_file


def save_json(filename, content):
    with open(filename, 'w') as stream:
        json.dump(content, stream, indent=4)


def parse_json_file(path, encoder_dict, convert_to='yolo', debug=False, original=False, mask=False):
    json_file = read_json(path)
    image_path = json_file.get('image_path')
    try:
        image_path = os.path.join(Path(path).parents[2], 'npz_json_png2',
                                  Path(image_path).stem, f'{Path(image_path).stem}.png')
    except:
        return
    if not os.path.exists(image_path):
        # print(f'Image by path {image_path} does not exists!')
        return

    # image_path, rectangles = get_data(json_file)

    converted_anno_dir_path = os.path.join(Path(image_path).parents[2], 'txt_from_json')

    if not Path(converted_anno_dir_path).exists():
        os.mkdir(converted_anno_dir_path)

    txt_file_path = f'{os.path.join(converted_anno_dir_path, Path(image_path).stem)}.txt'
    if mask:
        json_file_path = f'{os.path.join(converted_anno_dir_path, Path(image_path).stem)}.json'
    if os.path.exists(txt_file_path):
        if mask and os.path.exists(json_file_path):
            return image_path, txt_file_path, json_file_path
        else:
            return image_path, txt_file_path

    rectangles = []
    polygons = []
    if original:
        original_json_path = json_file.get('original_json_path').replace('npz_json_png', 'npz_json_png2')
        original_json_path = f'../../datasets{str(original_json_path)}'
        assert Path(original_json_path).exists(), f"File {original_json_path} does not exists!"
        json_file = read_json(original_json_path)
        objects = json_file.get('regions')

        for data_dict in objects:
            tnved = data_dict.get('tnved')
            user_role = data_dict.get('user_role')
            if tnved is not None and user_role == 'SPECIALIST':
                points = data_dict.get('points')
                coordinates = np.asarray(points)
                if coordinates.shape[0] < 3:
                    continue
                box = [
                    tnved, coordinates[:, 0].min(), coordinates[:, 1].min(), coordinates[:, 0].max(),
                    coordinates[:, 1].max()
                ]
                rectangles.append(box)
                polygons.append(points)
        rectangles = np.asarray(rectangles)
    else:
        objects = json_file.get('objects')
        rectangles, polygons = get_boxes(objects, return_polygons=True)

    if rectangles is None:
        return

    if debug:
        return rectangles

    if not os.path.exists(image_path):
        return

    if rectangles.size == 0:
        np.savetxt(txt_file_path, rectangles, delimiter=',', fmt='%s')  # X is an array
        if mask:
            save_json(json_file_path, [])
            return str(Path(image_path)), str(Path(txt_file_path)), str(Path(json_file_path))
        return str(Path(image_path)), str(Path(txt_file_path))

    else:
        w, h = Image.open(Path(image_path)).size
        rectangles[:, 0] = [encoder_dict.get('encoder')[i]for i in rectangles[:, 0]]

        tnved = rectangles[:, 0]
        boxes = rectangles[:, 1:].astype(np.float)

        yolo_boxes_x = (boxes[:, 2] + boxes[:, 0]) / 2
        yolo_boxes_y = (boxes[:, 3] + boxes[:, 1]) / 2

        yolo_boxes_w = (boxes[:, 2] - boxes[:, 0]).astype(np.float32)
        yolo_boxes_h = (boxes[:, 3] - boxes[:, 1]).astype(np.float32)

        yolo_boxes_x /= w
        yolo_boxes_y /= h
        yolo_boxes_w /= w
        yolo_boxes_h /= h

        rectangles = np.stack(
            [tnved, yolo_boxes_x, yolo_boxes_y, yolo_boxes_w, yolo_boxes_h]).transpose()

        np.savetxt(txt_file_path, rectangles, delimiter=' ', fmt='%s')
        if mask:
            save_json(json_file_path, polygons)

    if mask:
        return image_path, txt_file_path, json_file_path

    return image_path, txt_file_path


def main(path_):
    path_ = Path(path_)
    f = []
    f += glob.glob(str(path_ / '**' / '*.*'), recursive=True)
    f = [file_name for file_name in f if Path(file_name).suffix == '.json']

    lbl_dict = os.path.join(path_, 'lbl_dict.json')
    if lbl_dict in f:
        f.pop(f.index(lbl_dict))

    rect_list = {}
    for file in tqdm(f):
        rectangles = parse_json_file(file, debug=True)
        rect_list.update({file: rectangles})
    # remove zero shape arrays
    rect_values = [i for i in rect_list.values() if i.shape != (0,)]
    rect_values = np.vstack(rect_values)
    rect_values1 = rect_values[rect_values[:, 0] != None]
    tnved = list(set(rect_values1[:, 0]))
    tnved = sorted(tnved)
    label_dict = {key: i for i, key in enumerate(tnved)}
    labels = dict(labels=tnved, encoder=label_dict)
    with open(os.path.join(path_, 'lbl_dict.json'), 'w') as stream:
        json.dump(labels, stream, indent=4)

    # path = '../data/scantronic_tovary.yaml'
    # with open(path, 'r') as stream:
    #     data = yaml.safe_load(stream)
    # data.update(nc=len(tnved))
    # data.update(names=tnved)
    # with open(path, 'w') as stream:
    #     yaml.safe_dump(data, stream)


if __name__ == '__main__':
    path_w = 'C:/Users/D.Mihalkin/PycharmProjects/yolov5/dev/samples/train_sample'
    path_h = '/home/home/PycharmProjects/datasets/DATA2/Scans_2020-2021_2021-03-22/samples'
    path_s = ''
    if os.path.exists(path_w):
        main_path = path_w
    elif os.path.exists(path_h):
        main_path = path_h
    else:
        main_path = path_s
    main(main_path)
