import json
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import os


def te_wrapper(dictionary, key):
    try:
        return dictionary.get(key)
    except:
        raise ValueError(f'Dictionary does not contain {key} key.')


def get_preprocessed_jsons(json_file):
    original_json_path = te_wrapper(json_file, 'original_json_path')

    image_path = te_wrapper(json_file, 'image_path')
    objects = te_wrapper(json_file, 'objects')
    all_objects = []
    for key in objects.keys():
        current_objects = objects.get(key)

        if isinstance(current_objects, list):
            for i in range(len(current_objects)):
                if current_objects[i].__contains__('points'):
                    if not current_objects[i].__contains__('label'):
                        current_objects[i]['label'] = key
                    if not current_objects[i].__contains__('tnved'):
                        current_objects[i]['tnved'] = ''
                    all_objects.append(current_objects[i])

        elif isinstance(current_objects, dict):
            if current_objects.__contains__('points'):
                if not current_objects.__contains__('label'):
                    current_objects['label'] = key
                if not current_objects.__contains__('tnved'):
                    current_objects['tnved'] = ''
                all_objects.append(current_objects)
        else:
            raise NotImplemented

        # if current_objects is not None and current_objects != []:

    target_pack = []
    if len(all_objects) != 0:
        for entity in all_objects:
            coords = np.asarray(entity.get('points'))

            if entity.get('tnved') is None:
                continue

            try:
                xmin, ymin, xmax, ymax = coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()
            except:
                xmin, ymin, xmax, ymax = None, None, None, None

            if xmin is not None or ymin is not None or xmax is not None or ymax is not None:
                new_entity = np.array(
                    [
                        entity.get('label'),
                        entity.get('tnved'),
                        xmin, ymin,
                        xmax, ymax
                    ]
                )

            else:
                continue

            if new_entity is not None:
                target_pack.append(new_entity)

    return image_path, np.array(target_pack)


def get_original_jsons(json_file):
    object_list = json_file.get('regions')
    yolo_format = []
    for item in object_list:
        points = np.asarray(item.get('points'))  # x, y
        rect = np.asarray([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
        label = item.get('label')
        tnved = item.get('tnved')

        name, type_, tnved_codes = None, None, None
        if item.__contains__('category'):
            category = item.get('category')
            if category:
                if category.__contains__('name'):
                    name = category.get('name')
                if category.__contains__('type'):
                    type_ = category.get('type')
                if category.__contains__('tnved_codes'):
                    tnved_codes = category.get('tnved_codes')
        yolo_format.append(np.hstack([label, tnved, rect]))
    return np.asarray(yolo_format)


def get_data(json_file):
    if "regions" in json_file.keys():
        image_path = None
        rectangles = get_original_jsons(json_file)
    elif "objects" in json_file.keys():
        image_path, rectangles = get_preprocessed_jsons(json_file)
        if rectangles.size != 0:
            rectangles = rectangles[rectangles[:, 1] != '']
            rectangles = rectangles[(1 - np.isnan(rectangles[:, 1].astype(float))).astype(bool)]
    else:
        print(NotImplemented)
        pass
    return image_path, rectangles


def parse_json_file(path, convert_to='yolo', debug=False):
    try:
        with open(path, encoding='utf-8', mode='r') as stream:
            json_file = json.load(stream)
    except:
        with open(path, encoding='windows-1252', mode='r') as stream:  # utf-8
            json_file = json.load(stream)

    image_path, rectangles = get_data(json_file)

    image_path = os.path.join(Path(path).parents[2], 'npz_json_png2', Path(image_path).stem, Path(image_path).name)
    # if not Path(image_path).exists():
    #     image_path = image_path.replace('npz_json_png',
    #                                     'npz_json_png2')
    #     image_path = image_path.replace('/DATA2/Scans_2020-2021_2021-03-22/npz_json_png',
    #                                     'C:/Users/D.Mihalkin/PycharmProjects/yolov5/dev/samples/npz_json_png2')

    # if not Path(image_path).exists():
    image_path = image_path.replace('.npz', '.png')

    # converted_anno_dir_path = os.path.join(Path(image_path).parents[2], 'samples', 'txt_from_json')
    converted_anno_dir_path = os.path.join(Path(image_path).parents[2], 'txt_from_json')

    if not Path(converted_anno_dir_path).exists():
        os.mkdir(converted_anno_dir_path)

    txt_file_path = f'{os.path.join(converted_anno_dir_path, Path(image_path).stem)}.txt'
    if not Path(txt_file_path).exists():

        if convert_to == 'yolo':
            if rectangles.size == 0:
                np.savetxt(txt_file_path, rectangles, delimiter=',', fmt='%s')  # X is an array
                return str(Path(image_path)), str(Path(txt_file_path))

            try:
                w, h = Image.open(Path(image_path)).size
            except:
                return None
                # raise FileNotFoundError(f'{image_path}: file not found or corrupted!')

            if rectangles.size == 0:
                rectangles = np.array([])
            else:
                labels = rectangles[:, 0]
                tnved = rectangles[:, 1]

                boxes = rectangles[:, 2:].astype(np.float)

                yolo_boxes_x = (boxes[:, 2] + boxes[:, 0]) / 2
                yolo_boxes_y = (boxes[:, 3] + boxes[:, 1]) / 2

                yolo_boxes_w = (boxes[:, 2] - boxes[:, 0]).astype(np.float32)
                yolo_boxes_h = (boxes[:, 3] - boxes[:, 1]).astype(np.float32)

                yolo_boxes_x /= w
                yolo_boxes_y /= h
                yolo_boxes_w /= w
                yolo_boxes_h /= h

                rectangles = np.stack(
                    [tnved, yolo_boxes_x, yolo_boxes_y, yolo_boxes_w, yolo_boxes_h]).transpose() # labels,
                backup_ = rectangles.copy()
                with open(Path(os.path.join(Path(path).parents[1], 'lbl_dict.json')), mode='r') as stream:
                    lbl_dict = json.load(stream)
                try:
                    # rectangles[:, 0] = [int(lbl_dict.get('labels').get(lbl)) for lbl in rectangles[:, 0]]
                    rectangles[:, 0] = [int(lbl_dict.get('tnved').get(lbl)) for lbl in rectangles[:, 0]]
                    for i in range(rectangles[:, 0].size):
                        if rectangles[i, 0] is None:
                            rectangles[i, 0] = lbl_dict.get('tnved').get('null')
                    # eliminate objects with None tnved
                    rectangles = rectangles[rectangles[:, 0] != None]
                    rectangles = rectangles.astype(float)
                except:
                    pass
            try:
                np.savetxt(txt_file_path, rectangles, delimiter=' ', fmt='%s')  # X is an array
            except:
                pass
        else:
            raise Exception('WIP!')

    return str(Path(image_path)), str(Path(txt_file_path))


def main(path_):
    path_ = Path(path_)
    f = []
    f += glob.glob(str(path_ / '**' / '*.*'), recursive=True)
    f = [file_name for file_name in f if Path(file_name).suffix == '.json']
    try:
        f.pop(f.index(os.path.join(path_, 'lbl_dict.json')))
    except:
        pass
    rect_list = {}
    for file in f:
        rectangles = parse_json_file(file, debug=True)
        if rectangles is not None:
            rect_list[file] = rectangles
        # labels = list(set(all_boxes[:, 0]))
        # tnved = list(set(all_boxes[:, 1]))

    rez_rect_list = np.vstack(rect_list.values())
    labels = list(set(rez_rect_list[:, 0]))
    tnved = list(set(rez_rect_list[:, 1]))
    lbl_dict = dict(labels={lbl: idx for idx, lbl in enumerate(labels)},
                    tnved={lbl: idx for idx, lbl in enumerate(tnved)})
    save_name = Path(f'{Path(path, "lbl_dict")}.json')
    with open(save_name, encoding='utf-8', mode='w') as stream:
        json.dump(obj=lbl_dict, fp=stream, indent=4)

    for key in rect_list:
        try:
            rect_list[key][:, 0] = [lbl_dict.get('labels').get(lbl) for lbl in rect_list[key][:, 0]]
        except:
            pass
        rect_list[key][:, 1] = [lbl_dict.get('tnved').get(lbl) for lbl in rect_list[key][:, 1]]
        rect_list[key] = rect_list[key].astype(float)
        # with open(file, mode='r') as stream:
        np.savetxt('.txt', rect_list[key], delimiter=',')

    pass


if __name__ == '__main__':
    path_w = 'C:/Users/D.Mihalkin/PycharmProjects/yolov5/dev/samples/train_sample'
    path_h = '/home/home/PycharmProjects/datasets/DATA2/Scans_2020-2021_2021-03-22/samples'
    path_s = ''
    if os.path.exists(path_w):
        path = path_w
    elif os.path.exists(path_h):
        path = path_h
    else:
        path = path_s
    main(path)
