configs = {
    {
        'key': 'working_dir',
        'type': str,
        'default': '../converter/model_forge/f2e4a3a6-f9d7-49fc-a9da-79fb325c3899',
        'help': 'dir to store model artifacts'
    },
    {
        'key': 'name',
        'type': str,
        'default': '58d3ebdb-ba6c-4bec-a9fb-66195abb7f00',
        'help': 'dir to store experiment artifacts'
    },
    {
        'key': 'n_classes',
        'type': int,
        'default': 2,
        'help': 'Number of classes'
    },
    {
        'key': 'class_names',
        'type': list,
        'default': ['background', 'cab'],
        'help': 'class names'
    },
    {
        'key': 'inp-channels',
        'type': int,
        'default': 1,
        'help': 'Number of input channels'
    },
    {
        'key': 'epoch',
        'type': int,
        'default': 5,
        'range': [5, 300],
        'help': 'num epochs to train'
    },
    {
        'key': 'batch-size',
        'type': int,
        'default': 3,
        'range': [1, 'inf'],
        'help': 'num images per batch'
    },
    {
        'key': 'img-size',
        'type': list,
        'default': [1500, 300],
        'range': [224-2000, 224-2000],
        'help': 'image width - height'
    },
    {
        'key': 'learning-rate',
        'type': float,
        'default': 1e-5,
        'range': [1e-2, 1e-7],
        'help': 'learning rate'
    },
    # {
    #     'key': 'load',
    #     'type': str,
    #     'default': False,
    #     'help': 'weights path'
    # },
    {
        'key': 'scale',
        'type': float,
        'default': 1.0,
        'range': [0.5, 1.0],
        'help': 'Downscaling factor of the images'
    },
    # {
    #     'key': 'validation',
    #     'type': float,
    #     'default': 10.0,
    #     'range': [0.0, 100.0],
    #     'help': 'Percent of the data that is used as validation (0-100)'
    # },
    {
        'key': 'amp',
        'type': bool,
        'default': False,
        'help': 'Use mixed precision'
    },
    {
        'key': 'bilinear',
        'type': bool,
        'default': False,
        'help': 'Use bilinear upsampling'
    }
}
