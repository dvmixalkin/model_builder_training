import argparse
import logging
import sys
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet


def create_dataset_v1(dir_img, dir_mask, img_scale):
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
    return dataset


def create_dataset_v2(dir_img_, dir_mask_, pickle_data, img_size=[1500, 300]):
    from utils.scantronic.scantronic import MultiLabelScantronic
    dataset = MultiLabelScantronic(
        dir_img_, dir_mask_, pickle_data, img_size=img_size,
        suffix='_machine_mask', data_ext='.png'
    )
    return dataset


def create_dataset(dir_img_, dir_mask_, pickle_data, version=1, img_scale=1., img_size=[1500, 300]):
    if version == 1:
        return create_dataset_v1(dir_img_, dir_mask_, img_scale)
    elif version == 2:
        return create_dataset_v2(dir_img_, dir_mask_, pickle_data, img_size=img_size)
    else:
        raise NotImplemented


def create_splits(dataset, val_percent):
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    return {'len': n_train, 'set': train_set}, {'len': n_val, 'set': val_set}


def create_dataloaders(train_set, val_set, batch_size, num_workers):
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    return train_loader, val_loader


def train_net(net,
              device,
              start_epoch: int = 0,
              epochs: int = 5,
              img_size: list = None,  # Width, Height
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              # val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              num_workers: int = 0,  # @TODO change back to 4 num workers
              working_dir: str = None):

    parent_dir = Path(working_dir).parent
    dir_img = Path(f'{parent_dir}/train_target_files')
    dir_mask = Path(f'{parent_dir}/annotations')
    dir_checkpoint = Path(f'{working_dir}/weights/')
    pickle_file = f'{parent_dir}/train_dataset.pkl'
    with open(pickle_file, 'rb') as stream:
        annotation_data = pickle.load(stream)

    # 1. Create dataset
    if img_size is None:
        img_size = [1500, 300]

    # =================================--> WORKING_AREA <--=============================================================
    def manage_classes(annotation_data):
        machine_mapper = annotation_data['configs']['machine_mapper']
        if 'background' not in machine_mapper:
            if min(machine_mapper.values()) == 0:
                machine_mapper = {k:v+1 for k, v in machine_mapper.items()}
                machine_mapper['background'] = 0

        human_mapper = annotation_data['configs']['human_mapper']

        for keys in list(set(human_mapper.values())):


        annotation_data['configs']['machine_mapper']

        return machine_mapper
    managed_classes = manage_classes(annotation_data)
    # =================================--> WORKING_AREA <--=============================================================

    train_set = create_dataset(
        dir_img, dir_mask, pickle_data=annotation_data['train'],
        version=2, img_scale=1., img_size=img_size
    )
    val_set = create_dataset(
        dir_img, dir_mask, pickle_data=annotation_data['val'],
        version=2, img_scale=1., img_size=img_size
    )

    # 2. Split into train / validation partitions
    # train_set, val_set = create_splits(dataset, val_percent)

    # 3. Create data loaders
    train_loader, val_loader = create_dataloaders(train_set, val_set, batch_size, num_workers)

    # (Initialize logging)
    experiment = wandb.init(dir=f'{parent_dir}/train_model', project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_set.image_paths)}
        Validation size: {len(val_set.image_paths)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    epoch_score = 0
    for epoch in range(start_epoch, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train_set.image_paths), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                multiplier = 10  # @TODO change back to 10
                division_step = (len(train_set.image_paths) // (multiplier * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            # get val_score for epoch net
            score = evaluate(net, val_loader, device)
            logging.info('Validation Dice score: {}'.format(score))
            checkpoint_struct = {
                'img_size': img_size,
                'input_channels': net.n_channels,
                'class_names': net.class_names,
                'epoch': epoch,
                'dice_score': score,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'net': net.state_dict()
            }
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            # torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            torch.save(checkpoint_struct, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            if score >= epoch_score:
                epoch_score = score
                torch.save(checkpoint_struct, str(dir_checkpoint / 'best_dice.pth'.format(epoch)))

            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--working-dir', type=str,
                        default='../converter/model_forge/f2e4a3a6-f9d7-49fc-a9da-79fb325c3899',
                        help='dir to store model artifacts')
    parser.add_argument('--name', type=str, default='', help='Name of experiment')
    parser.add_argument('--names', type=list, default=['background', 'cab'], help='Class names')

    parser.add_argument('--inp-channels', type=int, default=1, help='Number of input channels')
    # parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--resume', type=bool, default=True, help='Load model from a .pth file')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=3, help='Batch size')
    parser.add_argument('--img-size', type=list, default=[1500, 300], help='image input size')

    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
    #                     help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    assert args.working_dir is not None, 'please, specify working dir.'

    from conveer.opt_checker import check_opts
    import yaml
    # get name to find latest run
    with open(f'{args.working_dir}/train_settings.yaml') as f:
        customer_cfgs = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))
    args, unmatched_configs = check_opts(opt=args, custom_cfg=vars(customer_cfgs))

    name_list = ['background']
    name_list.extend(args.names)
    args.names = name_list
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=args.inp_channels, n_classes=len(args.names), bilinear=args.bilinear)
    net.class_names = args.names
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')
    working_dir = f'{args.working_dir}/{args.name}'
    args.start_epoch = 0
    if args.resume:
        import glob
        weights_path = f'{working_dir}/weights'
        file_paths = glob.glob(f'{weights_path}/checkpoint_epoch*.pth')
        indexes = [int(str(Path(f).stem).strip('checkpoint_epoch')) for f in file_paths]
        try:
            last_weight_path = file_paths[indexes.index(max(indexes))]
            checkpoint = torch.load(last_weight_path, map_location='cpu')

            args.img_size = checkpoint['img_size']
            args.start_epoch = checkpoint['epoch']
            args.lr = checkpoint['learning_rate']
            assert checkpoint['input_channels'] == args.inp_channels, 'Channels num mismatch'

            if checkpoint['class_names'] != net.class_names:
                # @TODO weights = weights_manager(in_channels, classes)
                raise NotImplemented('model class_names mismatch')
            net.load_state_dict(checkpoint['net'])
            logging.info(f'Model loaded from {last_weight_path}')

        except Exception as e:
            print(f'{e} - No weights found to resume training')

    net.to(device=device)
    try:
        train_net(net=net,
                  start_epoch=args.start_epoch,
                  epochs=args.epochs,
                  img_size=args.img_size,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  # val_percent=args.val / 100,
                  amp=args.amp,
                  working_dir=working_dir)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
