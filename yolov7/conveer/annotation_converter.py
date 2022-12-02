import json
import os
from pathlib import Path
import numpy as np
from PIL import Image


def read_json(path):
    try:
        with open(path, encoding='utf-8', mode='r') as stream:
            json_file = json.load(stream)
    except:
        with open(path, encoding='windows-1252', mode='r') as stream:  # utf-8
            json_file = json.load(stream)
    return json_file


def get_boxes(objects, return_polygons=False, type_=None, cls_mark=None):
    objects_stack = []
    rectangles = []
    if return_polygons:
        polygons = []

    if type_ == 'carcass' or type_ == 'carcassSingle':
        objects_stack = objects['trailer']
    else:
        for part in objects:
            items = objects.get(part)
            if isinstance(items, list):
                tmp = [i for i in items if i.__contains__('tnved') and i.get('tnved') is not None]
                if tmp:
                    objects_stack.extend(tmp)
            else:
                if items.__contains__('tnved'):
                    objects_stack.append(items)

    for item in objects_stack:
        if cls_mark is None:
            lbl = item.get('tnved') if type_ == 'goods' else item.get('label')
        else:
            lbl = cls_mark
        # tnved = item.get('tnved')
        coordinates = item.get('points')
        np_coordinates = np.asarray(coordinates)
        box = [
            lbl,
            np_coordinates[:, 0].min(), np_coordinates[:, 1].min(),
            np_coordinates[:, 0].max(), np_coordinates[:, 1].max()
        ]
        rectangles.append(box)
        if return_polygons:
            polygons.append(coordinates)
    if return_polygons:
        return np.asarray(rectangles), polygons
    return np.asarray(rectangles)


def save_json(filename, content):
    with open(filename, 'w') as stream:
        json.dump(content, stream, indent=4)


def parse_json_file(path, encoder_dict, convert_to='yolo', debug=False, original=False, mask=False, type_=None,
                    additional_cls_mapper=None):
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
        cls_mark = None
        if additional_cls_mapper is not None:
            key = Path(path).stem.strip('_predict')
            cls_mark = additional_cls_mapper[key]
            # if cls_mark == 1:
            #     return
        rectangles, polygons = get_boxes(objects, return_polygons=True, type_=type_,
                                         cls_mark=cls_mark)

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
        tmp_array = []
        for i in rectangles[:, 0]:
            try:
                label = encoder_dict.get('encoder')[i]
            except:
                label = i
            tmp_array.append(label)
        rectangles[:, 0] = tmp_array
        del tmp_array

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
