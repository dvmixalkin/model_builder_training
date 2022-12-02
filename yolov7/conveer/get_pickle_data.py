import os
import numpy as np
from pathlib import Path


def adapt_paths(path, files_list, target_folder):  # ='train_target_files'
    files_list = [
        os.path.join(Path(path).parent, target_folder, str(Path(file).name))
        for file in files_list
    ]
    return files_list


def make_yolo_annotations(img_files, label_files, pickled_annotation_data):
    for image, lbls in zip(img_files, label_files):
        np_array = np.array(pickled_annotation_data[image])
        np.savetxt(lbls, np_array, delimiter=' ', fmt='%s')
