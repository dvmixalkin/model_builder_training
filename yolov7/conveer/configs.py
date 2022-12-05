all_configs = {
    'general_settings': [
        {
            'key': 'weights',
            'type': str,
            'default': 'yolov7.pt',
            'help': 'initial weights path'
        },
        {
            'key': 'cfg',
            'type': str,
            'default': '',
            'help': 'model.yaml path'
        },
        {
            'key': 'epochs',
            'type': int,
            'default': 300,
            'help': 'num epochs'
        },
        {
            'key': 'batch-size',
            'type': int,
            'default': 1,
            'help': 'total batch size for all GPUs'
        },
        {
            'key': 'img-size',
            'type': int or list,
            'default': [640, 640],
            'help': '[train, test] image sizes'
        },
        {
            'key': 'resume',
            'type': bool,
            'default': True,
            'help': 'resume most recent training'
        },
        {
            'key': 'nosave',
            'type': bool,
            'default': True,
            'help': 'only save final checkpoint'
        },
        {
            'key': 'notest',
            'type': bool,
            'default': True,
            'help': 'only test final epoch'
        },
        {
            'key': 'device',
            'type': str,
            'default': 'cpu',
            'help': 'cuda device, i.e. 0 or 0,1,2,3 or cpu'
        },
        {
            'key': 'project',
            'type': str,
            'default': 'runs/train',
            'help': 'save to project/name'
        },
        # {
        #     'key': 'entity',
        #     'type': '',  # @ TODO specify type (str)
        #     'default': None,
        #     'help': 'W&B entity'
        # },
        {
            'key': 'name',
            'type': str,
            'default': 'exp',
            'help': 'save to project/name'
        },
        {
            'key': 'exist-ok',
            'type': bool,
            'default': True,
            'help': 'existing project/name ok, do not increment'
        },
        {
            'key': 'save_period',
            'type': int,
            'default': -1,
            'help': 'Log model after every "save_period" epoch'
        },
        # {
        #     'key': 'artifact_alias',
        #     'type': str,
        #     'default': 'latest',
        #     'help': 'version of dataset artifact to be used'
        # },
        {
            'key': 'freeze',
            'type': int or list,
            'default': [0],
            'help': 'Freeze layers: backbone of yolov7=50, first3=0 1 2'
        },
        # {
        #     'key': 'v5-metric',
        #     'type': bool,
        #     'default': True,
        #     'help': 'assume maximum recall as 1.0 in AP calculation'
        # },
    ],
    'model_settings': [
        {
            'key': 'noautoanchor',
            'type': bool,
            'default': True,
            'help': 'disable autoanchor check'
        },
        {
            'key': 'sync-bn',
            'type': bool,
            'default': True,
            'help': 'use SyncBatchNorm, only available in DDP mode'
        },
        {
            'key': 'anchor_t',
            'type': float,
            'default': 4.0,
            'range': [2.0, 8.0],
            'help': 'anchor-multiple threshold'
        },
        {
            'key': 'anchors',
            'type': int,
            'default': 3,
            'range': [2, 10],
            'help': 'anchors per output grid (0 to ignore)'
        },
    ],
    # 'hyperparameters': [
    #     {
    #         'key': 'hyp',
    #         'type': str,
    #         'default': 'data/hyp.scratch.p5.yaml',
    #         'help': 'hyperparameters path'
    #     },
    #     {
    #         'key': 'evolve',
    #         'type': bool,
    #         'default': True,
    #         'help': 'evolve hyperparameters'
    #     },
    # ],
    'dataset': [
        {
            'key': 'data',
            'type': str,
            'default': 'data/coco_small.yaml',
            'help': 'data.yaml path'
        },
        {
            'key': 'rect',
            'type': bool,
            'default': True,  # action='store_true',
            'help': 'rectangular training'
        },
        {
            'key': 'cache-images',
            'type': bool,
            'default': True,
            'help': 'cache images for faster training'
        },
        {
            'key': 'image-weights',
            'type': bool,
            'default': True,
            'help': 'use weighted image selection for training'
        },
        {
            'key': 'multi-scale',
            'type': bool,
            'default': True,
            'help': 'vary img-size +/- 50%%'
        },
        {
            'key': 'single-cls',
            'type': bool,
            'default': False,
            'help': 'train multi-class data as single-class'
        },
        {
            'key': 'workers',
            'type': int,
            'default': 8,
            'help': 'maximum number of dataloader workers'
        },
        {
            'key': 'quad',
            'type': bool,
            'default': True,
            'help': 'quad dataloader'
        },
        {
            'key': 'hsv_h',
            'type': float,
            'default': 0.015,
            'range': [0.0, 0.1],
            'help': 'image HSV-Hue augmentation (fraction)'
        },
        {
            'key': 'hsv_s',
            'type': float,
            'default': 0.7,
            'range': [0.0, 0.9],
            'help': 'image HSV-Saturation augmentation (fraction)'
        },
        {
            'key': 'hsv_v',
            'type': float,
            'default': 0.4,
            'range': [0.0, 0.9],
            'help': 'image HSV-Value augmentation (fraction)'
        },
        {
            'key': 'degrees',
            'type': float,
            'default': 0.0,
            'range': [0.0, 45.0],
            'help': 'image rotation (+/- deg)'
        },
        {
            'key': 'translate',
            'type': float,
            'default': 0.2,
            'range': [0.0, 0.9],
            'help': 'image translation (+/- fraction)'
        },
        {
            'key': 'scale',
            'type': float,
            'default': 0.5,
            'range': [0.0, 0.9],
            'help': 'image scale (+/- gain)'
        },
        {
            'key': 'shear',
            'type': float,
            'default': 0.0,
            'range': [0.0, 10.0],
            'help': 'image shear (+/- deg)'
        },
        {
            'key': 'perspective',
            'type': float,
            'default': 0.0,
            'range': [0.0, 0.001],
            'help': 'image perspective (+/- fraction), range 0-0.001'
        },
        {
            'key': 'flipud',
            'type': float,
            'default': 0.0,
            'range': [0.0, 1.0],
            'help': 'image flip up-down (probability)'
        },
        {
            'key': 'fliplr',
            'type': float,
            'default': 0.5,
            'range': [0.0, 1.0],
            'help': 'image flip left-right (probability)'
        },
        {
            'key': 'mosaic',
            'type': float,
            'default': 1.0,
            'range': [0.0, 1.0],
            'help': 'image mixup (probability)'
        },
        {
            'key': 'mixup',
            'type': float,
            'default': 0.0,
            'range': [0.0, 1.0],
            'help': 'image mixup (probability)'
        },
    ],
    'optimizer': [
        {
            'key': 'adam',
            'type': bool,
            'default': True,
            'help': 'use torch.optim.Adam() optimizer'
        },
        {
            'key': 'linear-lr',
            'type': bool,
            'default': True,
            'help': 'linear LR'
        },
        {
            'key': 'lr0',
            'type': float,
            'default': 0.01,
            'range': [1e-5, 1e-1],
            'help': 'initial learning rate (SGD=1E-2, Adam=1E-3)'
        },
        {
            'key': 'lrf',
            'type': float,
            'default': 0.1,
            'range': [1e-2, 1],
            'help': 'final OneCycleLR learning rate (lr0 * lrf)'
        },
        {
            'key': 'momentum',
            'type': float,
            'default': 0.937,
            'range': [0.6, 0.98],
            'help': 'SGD momentum/Adam beta1'
        },
        {
            'key': 'weight_decay',
            'type': float,
            'default': 0.0005,
            'range': [0.0, 0.001],
            'help': 'optimizer weight decay'
        },
        {
            'key': 'warmup_epochs',
            'type': int,
            'default': 3,
            'range': [0, 5],
            'help': 'warmup epochs (fractions ok)'
        },
        {
            'key': 'warmup_momentum',
            'type': float,
            'default': 0.8,
            'range': [0.0, 0.95],
            'help': 'warmup initial momentum'
        },
        {
            'key': 'warmup_bias_lr',
            'type': float,
            'default': 0.1,
            'range': [0.0, 0.2],
            'help': 'warmup initial bias lr'
        },
    ],
    'loss_fns': [
        {
            'key': 'box',
            'type': float,
            'default': 0.05,
            'range': [0.02, 0.2],
            'help': 'box loss gain'
        },
        {
            'key': 'cls',
            'type': float,
            'default': 0.3,
            'range': [0.2, 4.0],
            'help': 'cls loss gain'
        },
        {
            'key': 'cls_pw',
            'type': float,
            'default': 1.0,
            'range': [0.5, 2.0],
            'help': 'cls BCELoss positive_weight'
        },
        {
            'key': 'obj',
            'type': float,
            'default': 0.7,
            'range': [0.2, 4.0],
            'help': 'obj loss gain (scale with pixels)'
        },
        {
            'key': 'obj_pw',
            'type': float,
            'default': 1.0,
            'range': [0.5, 2.0],
            'help': 'obj BCELoss positive_weight'
        },
        {
            'key': 'iou_t',
            'type': float,
            'default': 0.2,
            'range': [0.1, 0.7],
            'help': 'IoU training threshold'
        },
        {
            'key': 'fl_gamma',
            'type': float,
            'default': 0.0,
            'range': [0.0, 2.0],
            'help': 'focal loss gamma (efficientDet default gamma=1.5)'
        },
    ],
    'others': [
        # {
        #     'key': 'bucket',
        #     'type': str,
        #     'default': '',
        #     'help': 'gsutil bucket'
        # },
        {
            'key': 'local_rank',
            'type': int,
            'default': -1,
            'help': 'DDP parameter, do not modify'
        },
        {
            'key': 'label-smoothing',
            'type': float,
            'default': 0.0,
            'help': 'Label smoothing epsilon'
        },
        # {
        #     'key': 'upload_dataset',
        #     'type': bool,
        #     'default': True,
        #     'help': 'Upload dataset as W&B artifact table'
        # },
        {
            'key': 'bbox_interval',
            'type': int,
            'default': -1,
            'help': 'Set bounding-box image logging interval for W&B'
        },
        {
            'key': 'copy_paste',
            'type': float,
            'default': 0.0,
            'range': [0.0, 1.0],
            'help': 'segment copy-paste (probability)'
        },
        {
            'key': 'paste_in',
            'type': float,
            'default': 0.0,
            'range': [0.0, 1.0],
            'help': 'segment copy-paste (probability)'
        },
    ]
}
