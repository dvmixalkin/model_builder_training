import os
import argparse
import sys
import yaml
from pathlib import Path
import torch
import glob
import torch.distributed as dist
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.callbacks import Callbacks
from utils.torch_utils import select_device  # , EarlyStopping, ModelEMA, de_parallel, torch_distributed_zero_first
from utils.general import (LOGGER, print_args, check_requirements, get_latest_run, check_file, check_yaml, colorstr,
                           methods, init_seeds, check_dataset, check_suffix, intersect_dicts, check_amp)
# from utils.general import (, , , check_git_status, check_img_size,
#                            , check_version,
#                            , , labels_to_class_weights,
#                            labels_to_image_weights, , one_cycle, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from yolov5.conveer.opt_checker import check_opts, get_configs
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from models.yolo import Model

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/scantronic_tovary.yaml', help='dataset.yaml path')

    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=True, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def train_resume(opt):
    search_dir = f'{opt.project}/train_model/{opt.name}'
    ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run(
        search_dir=search_dir)  # specified or most recent path
    assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
    with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))  # replace
    opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
    LOGGER.debug(f'Resuming training from {ckpt}')
    return opt


def train_from_scratch(opt):
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    if opt.name == 'cfg':
        opt.name = Path(opt.cfg).stem  # use model.yaml as name
    opt.save_dir = str(Path(Path(opt.project) / opt.name))
    return opt


class YOLOv5ToolBox:
    def __init__(self, uuid=0, data_path=None):
        self.uuid = uuid
        self.callbacks = Callbacks()
        self.opt = parse_opt()
        self.plots = not self.opt.evolve and not self.opt.noplots
        self.device = select_device(self.opt.device, batch_size=self.opt.batch_size)
        self.cuda = self.device.type != 'cpu'
        init_seeds(1 + RANK)
        # ==========================
        self.model = None
        self.weights_dir = None
        self.last_path = None
        self.best_path = None
        self.ckpt = None
        self.hyp = None
        self.unmatched_configs = None
        self.nc = None
        self.names = None

    def hyper_parameters(self):
        if isinstance(self.opt.hyp, str):
            with open(self.opt.hyp, errors='ignore') as f:
                self.hyp = yaml.safe_load(f)  # load hyps dict

        self.hyp, self.unmatched_configs = check_opts(self.hyp, self.unmatched_configs)
        LOGGER.debug(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in self.hyp.items()))

    def directories(self):  # save_dir
        self.weights_dir = Path(self.opt.save_dir) / 'weights'  # weights dir
        self.weights_dir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last_path = self.weights_dir / 'last.pt'
        self.best_path = self.weights_dir / 'best.pt'

    def save_run_settings(self):
        with open(self.opt.save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(self.hyp, f, sort_keys=False)
        with open(self.opt.save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(self.opt), f, sort_keys=False)

    def load_pretrained(self, weights, nc, resume):
        self.ckpt = torch.load(weights, map_location='cpu')
        self.model = Model(self.opt.cfg or self.ckpt['model'].yaml, ch=3, nc=nc, anchors=self.hyp.get('anchors'))
        exclude = ['anchor'] if (self.opt.cfg or self.hyp.get('anchors')) and not resume else []
        csd = self.ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, self.model.state_dict(), exclude=exclude)
        self.model.load_state_dict(csd, strict=False)
        LOGGER.debug(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from {weights}')

    def load_from_scratch(self, nc):
        model = Model(self.opt.cfg, ch=3, nc=nc, anchors=self.hyp.get('anchors'))  # .to(device)  # create
        return model

    def prepare_model(self, pretrained):
        if pretrained:
            epoch_nums = []
            checkpoints = glob.glob(str(self.last_path.parents[2]) + '/**/weights/last.pt')
            if len(checkpoints) > 0:
                for ckpt in checkpoints:
                    v = (ckpt.split('/')[-3][len(self.opt.name):])
                    if v == '':
                        v = 1
                    epoch_nums.append(int(v))
                last_epoch_idx = epoch_nums.index(max(epoch_nums))
                weights = checkpoints[last_epoch_idx]
                self.load_pretrained(weights, self.nc, self.opt.resume)
            else:
                self.load_from_scratch(self.nc)
        else:
            self.load_from_scratch(self.nc)

        self.model = self.model.to(self.device)
        self.amp = check_amp(self.model)
        return self.model, self.amp

    def freeze_layers(self):
        # Freeze
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                LOGGER.debug(f'freezing {k}')
                v.requires_grad = False

    def init_optimizer(self):
        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        LOGGER.debug(f"Scaled weight_decay = {hyp['weight_decay']}")

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        if opt.optimizer == 'Adam':
            optimizer = Adam(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        elif opt.optimizer == 'AdamW':
            optimizer = AdamW(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = SGD(g[2], lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': g[0], 'weight_decay': hyp['weight_decay']})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)
        LOGGER.debug(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                     f"{len(g[1])} weight (no decay), {len(g[0])} weight, {len(g[2])} bias")
        del g

    def init_scheduler(self):
        if opt.cos_lr:
            lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        else:
            lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        plot_lr_scheduler(optimizer, scheduler, epochs)

    def resume_training(self):
        start_epoch, best_fitness = 0, 0.0
        if pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if ema and ckpt.get('ema'):
                csd = ckpt['ema'].float().state_dict()
                csd = intersect_dicts(csd, ema.ema.state_dict(), exclude=exclude)
                ema.ema.load_state_dict(csd, strict=False)
                ema.updates = ckpt['updates']

            # Epochs
            start_epoch = ckpt['epoch'] + 1
            # if resume:
            #     assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
            if epochs < start_epoch:
                LOGGER.debug(
                    f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
                epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, csd

    def check_img_size(self):
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    def check_train_batch_size(self):
        if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
            batch_size = check_train_batch_size(model, imgsz, amp)
            loggers.on_params_update({"batch_size": batch_size})

    def dp_model(self):
        if cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                           'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            model = torch.nn.DataParallel(model)

    def ddp_mode(self):
        if cuda and RANK != -1:
            if check_version(torch.__version__, '1.11.0'):
                model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
            else:
                model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    def sync_batch_norm(self):
        if opt.sync_bn and cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            LOGGER.debug('Using SyncBatchNorm()')

    def train_dataloader(self):
        train_loader, dataset = create_dataloader(
            train_path,
            imgsz,
            batch_size // WORLD_SIZE,
            gs,
            single_cls,
            hyp=hyp,
            augment=True,
            cache=None if opt.cache == 'val' else opt.cache,
            rect=opt.rect,
            rank=LOCAL_RANK,
            workers=workers,
            image_weights=opt.image_weights,
            quad=opt.quad,
            prefix=colorstr('train: '),
            shuffle=True
        )

    def val_dataloader(self):
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

    def model_atributes(self):
        nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        model.names = names

    def start_training(self):
        t0 = time.time()
        nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        stopper = EarlyStopping(patience=opt.patience)
        compute_loss = ComputeLoss(model)  # init loss class
        callbacks.run('on_train_start')
        LOGGER.debug(f'Image sizes {imgsz} train, {imgsz} val\n'
                     f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                     f"Logging results to {colorstr('bold', save_dir)}\n"
                     f'Starting training for {epochs} epochs...')

    def single_epoch_train(self):
        for i, (imgs, targets, paths, _) in pbar:  # batch ---------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi,
                                        [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in
                          imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots)
                if callbacks.stop_training:
                    return
            trn_imgs_processed += len(paths)
            LOGGER.info(json.dumps({"current_epoch": epoch,
                                    "epochs_total": epochs,
                                    "processed_images": trn_imgs_processed + val_imgs_processed,
                                    "total_images": (trn_loader_imgnum + val_loader_imgnum)}))
            # end batch ----------------------------------------------------------------------------------------

    def train(self):  # hyp is path/to/hyp.yaml or hyp dictionary
        self.callbacks.run('on_pretrain_routine_start')

        # self.opt.save_dir,   self.opt.epochs, self.opt.batch_size, self.opt.weights,
        # self.opt.single_cls, self.opt.evolve, self.opt.data,       self.opt.cfg,
        # self.opt.resume,     self.opt.noval,  self.opt.nosave,     self.opt.workers,
        # self.opt.freeze

        # Directories: SET weights_dir, last_path, best_path
        self.directories()

        # Get Hyper parameters
        self.hyper_parameters()

        # Save run settings
        self.save_run_settings()

        # Loggers
        if RANK in {-1, 0}:
            # loggers instance
            loggers = Loggers(Path(self.opt.save_dir), self.opt.weights, self.opt, self.hyp, LOGGER)

            # Register actions
            for k in methods(loggers):
                self.callbacks.register_action(k, callback=getattr(loggers, k))

        # check if None
        data_dict = None
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = data_dict or check_dataset(self.opt.data)

        data_dict, unmatched_configs = check_opts(data_dict, self.unmatched_configs)
        train_path, val_path = data_dict['train'], data_dict['val']

        # number of classes
        self.nc = 1 if self.opt.single_cls else int(data_dict['nc'])

        # class names
        self.names = ['item'] if self.opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']

        # verify classes and number classes
        assert len(self.names) == self.nc, f'{len(self.names)} names found for nc={self.nc} dataset in {self.opt.data}'

        # Model
        check_suffix(self.opt.weights, '.pt')  # check weights
        pretrained = self.opt.weights.endswith('.pt')
        model = self.prepare_model(pretrained)

        self.freeze_layers()

        # Image size
        self.check_img_size()

        # Batch size
        self.check_train_batch_size()

        # get optimizer
        self.init_optimizer()

        # get scheduler
        self.init_scheduler()

        # EMA
        ema = ModelEMA(model) if RANK in {-1, 0} else None

        # Resume
        self.resume_training()

        # DP mode
        self.dp_model()

        # SyncBatchNorm
        self.sync_batch_norm()

        # Trainloader

        workers = 0
        self.train_dataloader()

        mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
        nb = len(train_loader)  # number of batches
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

        # Process 0
        if RANK in {-1, 0}:
            self.val_dataloader()
            if not resume:
                labels = np.concatenate(dataset.labels, 0)
                if plots:
                    plot_labels(labels, names, save_dir)

                # Anchors
                if not opt.noautoanchor:
                    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end')

        # DDP mode
        self.ddp_mode()

        # Model attributes
        self.model_atributes()

        # Start training
        self.start_training()

        # for epoch in range(start_epoch,
        #                    epochs):  # epoch ------------------------------------------------------------------
        #
        #     trn_loader_imgnum, val_loader_imgnum = len(train_loader.dataset), len(val_loader.dataset)
        #     trn_imgs_processed = 0
        #     val_imgs_processed = 0
        #
        #     callbacks.run('on_train_epoch_start')
        #     model.train()
        #
        #     # Update image weights (optional, single-GPU only)
        #     if opt.image_weights:
        #         cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
        #         iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
        #         dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        #
        #     # Update mosaic border (optional)
        #     # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        #     # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
        #
        #     mloss = torch.zeros(3, device=device)  # mean losses
        #     if RANK != -1:
        #         train_loader.sampler.set_epoch(epoch)
        #     pbar = enumerate(train_loader)
        #     LOGGER.debug(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        #     if RANK in {-1, 0}:
        #         pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        #                     disable=False)  # True progress bar
        #     optimizer.zero_grad()
        #     self.single_epoch_train()

        #     # Scheduler
        #     lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        #     scheduler.step()
        #
        #     if RANK in {-1, 0}:
        #         # mAP
        #         callbacks.run('on_train_epoch_end', epoch=epoch)
        #         ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        #         final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
        #         if not noval or final_epoch:  # Calculate mAP
        #             results, maps, _ = val.run(data_dict,
        #                                        batch_size=batch_size // WORLD_SIZE * 2,
        #                                        imgsz=imgsz,
        #                                        model=ema.ema,
        #                                        single_cls=single_cls,
        #                                        dataloader=val_loader,
        #                                        save_dir=save_dir,
        #                                        plots=False,
        #                                        callbacks=callbacks,
        #                                        compute_loss=compute_loss)
        #             val_imgs_processed += val_loader_imgnum
        #         LOGGER.info(json.dumps({"current_epoch": epoch,
        #                                 "epochs_total": epochs,
        #                                 "processed_images": trn_imgs_processed + val_imgs_processed,
        #                                 "total_images": (trn_loader_imgnum + val_loader_imgnum)}))
        #         # Update best mAP
        #         fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        #         if fi > best_fitness:
        #             best_fitness = fi
        #         log_vals = list(mloss) + list(results) + lr
        #
        #         LOGGER.info(json.dumps({"current_epoch": epoch,
        #                                 "epochs_total": epochs,
        #                                 "train/box_loss": log_vals[0].item(),
        #                                 "train/obj_loss": log_vals[1].item(),
        #                                 "train/cls_loss": log_vals[2].item(),
        #                                 "metrics/precision": log_vals[3],
        #                                 "metrics/recall": log_vals[4],
        #                                 "metrics/mAP_0.5": log_vals[5],
        #                                 "metrics/mAP_0.5:0.95": log_vals[6],
        #                                 "val/box_loss": log_vals[7],
        #                                 "val/obj_loss": log_vals[8],
        #                                 "val/cls_loss": log_vals[9],
        #                                 "x/lr0": log_vals[10],
        #                                 "x/lr1": log_vals[11],
        #                                 "x/lr2": log_vals[12]}))
        #
        #         callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
        #
        #         # Save model
        #         if (not nosave) or (final_epoch and not evolve):  # if save
        #             save_model(
        #                 opt, last, best, model, ema, optimizer, epoch, imgsz, best_fitness, callbacks, w, fi,
        #                 final_epoch
        #             )
        #
        #         # Stop Single-GPU
        #         if RANK == -1 and stopper(epoch=epoch, fitness=fi):
        #             break
        #
        #     # end epoch ----------------------------------------------------------------------------------------------------
        # # end training -----------------------------------------------------------------------------------------------------
        # if RANK in {-1, 0}:
        #     LOGGER.debug(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        #     for f in last, best:
        #         if f.exists():
        #             strip_optimizer(f)  # strip optimizers
        #             if f is best:
        #                 LOGGER.debug(f'\nValidating {f}...')
        #                 results, _, _ = val.run(
        #                     data_dict,
        #                     batch_size=batch_size // WORLD_SIZE * 2,
        #                     imgsz=imgsz,
        #                     model=attempt_load(f, device).half(),
        #                     iou_thres=0.6,
        #                     single_cls=single_cls,
        #                     dataloader=val_loader,
        #                     save_dir=save_dir,
        #                     save_json=False,
        #                     verbose=True,
        #                     plots=plots,
        #                     callbacks=callbacks,
        #                     compute_loss=compute_loss)  # val best model with plots
        #
        #     callbacks.run('on_train_end', last, best, plots, epoch, results)
        #
        torch.cuda.empty_cache()
        return results

    def process(self, root_folder: str, train_name: str):
        path_to_data = os.path.join(root_folder, train_name)
        loggerName = train_name
        LOGGER.name = loggerName
        unmatched_configs = get_configs(f'{root_folder}/train_settings.yaml')
        unmatched_configs['path_to_data'] = path_to_data
        self.opt, self.unmatched_configs = check_opts(self.opt, unmatched_configs)
        print(self.opt)

        # Checks
        print_args(vars(self.opt))
        check_requirements(exclude=['thop'])

        # Try to Resume
        try:
            self.opt = train_resume(self.opt)
        # else:
        except Exception as e:
            print(e)
            self.opt = train_from_scratch(self.opt)

        self.train()

        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.debug('Destroying process group... ')
            dist.destroy_process_group()


def main():
    toolbox = YOLOv5ToolBox()
    root_folder = '../converter/model_forge/f2e4a3a6-f9d7-49fc-a9da-79fb325c3899'
    train_name = '58d3ebdb-ba6c-4bec-a9fb-66195abb7f00'
    toolbox.process(root_folder, train_name)


if __name__ == '__main__':
    main()
