import torch
import random
from pathlib import Path


def weights_manager(net, checkpoint):
    try:
        net_state_dict = net.state_dict().copy()
        if checkpoint['class_names'] != net.class_names:
            for key in net_state_dict.keys():
                if key in checkpoint['net']:
                    if net_state_dict[key].shape == checkpoint['net'][key].shape:
                        net_state_dict[key] = checkpoint['net'][key]
                    else:
                        print(key, net_state_dict[key].shape, checkpoint['net'][key].shape)

                        dims = len(net_state_dict[key].shape)
                        target_dim = 0
                        for dim in range(dims):
                            if net_state_dict[key].shape[dim] != checkpoint['net'][key].shape[dim]:
                                target_dim = dim
                                break
                        mode = None
                        condition = net_state_dict[key].shape[target_dim] > checkpoint['net'][key].shape[target_dim]
                        mode = 'add' if condition else 'remove'
                        new_weights = []
                        for idx, class_name in enumerate(net.class_names):
                            neuron_weight = net_state_dict[key].select(dim=target_dim, index=idx)
                            if class_name in checkpoint['class_names']:
                                new_weights.append(neuron_weight)
                            else:  # class_name NOT in checkpoint['class_names']
                                if mode == 'add':
                                    random_idx = random.randint(0, net_state_dict[key].shape[target_dim] - 1)
                                    w = net_state_dict[key].select(dim=target_dim, index=random_idx)
                                    neuron_weight = torch.nn.init.uniform_(w)

                                    new_weights.append(neuron_weight)
                                else:
                                    continue
                        net_state_dict[key] = torch.stack(new_weights, dim=target_dim)
        net.load_state_dict(net_state_dict)
        return net
    except Exception as e:
        print(f'{e} - No weights found to resume training')


def unet_manager(args, net, logging):
    import glob
    working_dir = f'{args.working_dir}/{args.name}'
    weights_path = f'{working_dir}/weights'
    file_paths = glob.glob(f'{weights_path}/checkpoint_epoch*.pth')
    indexes = [int(str(Path(f).stem).strip('checkpoint_epoch')) for f in file_paths]
    try:
        weight_path = file_paths[indexes.index(max(indexes))]
        checkpoint = torch.load(weight_path, map_location='cpu')

        if args.img_size != checkpoint['img_size']:
            logging.info(f'input image_size changed from {args.img_size} to {checkpoint["img_size"]}')
            args.img_size = checkpoint['img_size']
        args.start_epoch = checkpoint['epoch']
        # args.lr = checkpoint['learning_rate']
        assert checkpoint['input_channels'] == args.inp_channels, 'Channels num mismatch'

        net = weights_manager(net, checkpoint)
        logging.info(f'Model loaded from {weight_path}')
    except Exception as e:
        print(e)
    return net, args


def yolov5_manager(weights_path):
    # from yolov5 import *
    pass


def yolov7_manager(weights_path):

    pass


def main():
    in_channels, classes = 3, ['1', '2', '3']
    unet_manager(in_channels, classes)
    pass


if __name__ == '__main__':
    main()
