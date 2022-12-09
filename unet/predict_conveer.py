import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from pathlib import Path
from utils.scantronic.scantronic import mask2poly


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m',
                        # default='checkpoints/checkpoint_epoch50.pth',
                        default='checkpoints/best_dice.pth',
                        metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--img-size', type=list, default=[1500, 300], help='image input size')
    parser.add_argument('--inp-channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=False)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


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


def get_bbox(mask, threshold=100):
    polygons = mask2poly(mask)
    bbox_coordinates = []
    for polygon in polygons:
        if polygon.area < threshold:
            continue
        x, y = polygon.exterior.xy
        poly_coordinates = np.stack([x, y]).transpose()
        x_min, x_max = poly_coordinates[:, 0].min(), poly_coordinates[:, 0].max()
        y_min, y_max = poly_coordinates[:, 1].min(), poly_coordinates[:, 1].max()
        bbox_coordinates.append([x_min, y_min, x_max, y_max])
    return bbox_coordinates


def get_bboxes_from_mask(mask):
    all_cls_boxes = []
    for cls_idx, channel in enumerate(mask):
        if cls_idx == 0:
            continue
        print(channel.shape)
        boxes = np.array(get_bbox(channel))
        cls_info = np.hstack([np.ones(boxes.shape[0]).reshape(-1, 1), boxes]).astype(int)
        all_cls_boxes.append(cls_info)
    return np.vstack(all_cls_boxes)


def vis_boxes(image, coordinates):
    import cv2
    if not isinstance(image, np.ndarray):
        image = np.array(image.convert('RGB'))
    img_copy = image.copy()
    imgHeight, imgWidth, _ = img_copy.shape
    for bbox in coordinates:
        label, left, top, right, bottom = bbox
        color, thick = (255, 0, 0), 5
        cv2.rectangle(img_copy, (left, top), (right, bottom), color, thick)
        cv2.putText(img_copy, str(label), (left, top - 12), 0, 1e-3 * imgHeight, color, thick // 3)
    Image.fromarray(img_copy).show()


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    # in_files = [
    #     '../converter/model_forge/f2e4a3a6-f9d7-49fc-a9da-79fb325c3899/train_target_files/0e21bd92-3d51-4472-be5f-3a2ee4f17fa7_height_1345_width_5484.npy'
    # ]
    # args.input = in_files
    out_files = get_output_filenames(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.model, map_location=device)
    try:
        img_size = checkpoint['img_size']
        weights = checkpoint['net']
        input_channels = checkpoint['input_channels']
        classes = checkpoint['classes']

    except Exception as e:
        print(e)
        weights = checkpoint
        img_size = args.img_size
        input_channels = args.inp_channels
        classes = args.classes

    net = UNet(n_channels=input_channels, n_classes=classes, bilinear=args.bilinear)

    logging.info(f'Loading model {args.model}')
    net.load_state_dict(weights)

    logging.info(f'Using device {device}')
    net.to(device=device)
    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        # img = Image.open(filename)
        img = get_image(filename)
        img = img.resize(img_size, resample=Image.Resampling.BICUBIC)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # get bounding boxes from masks
        bboxes = get_bboxes_from_mask(mask)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
            vis_boxes(img, bboxes)
