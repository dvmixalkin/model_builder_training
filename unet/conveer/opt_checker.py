import os
import distutils


def parse_yaml(path):
    import yaml
    with open(path, 'r', encoding='UTF-8') as stream:
        configs = yaml.safe_load(stream)
    return configs


def get_configs(_cfg):
    if isinstance(_cfg, str):
        configs = parse_yaml(_cfg)
    elif isinstance(_cfg, dict):
        configs = _cfg
    else:
        raise NotImplemented
    return configs


def parse_list(opt, k, v):
    import json
    try:
        content_type = None

        if len(getattr(opt, k)) > 0:
            content_type = type(getattr(opt, k)[0])

        data = (json.loads(v) if isinstance(v, str) else v)
        value = [content_type(val) for val in data]
        return value
    except Exception as e:
        print(e)


def parse_types(opt, k, v):
    _type = type(getattr(opt, k))
    try:
        if _type == list:
            value = parse_list(opt, k, v)
        elif _type is type(None):
            value = v
        elif _type == bool:
            value = bool(distutils.util.strtobool(v))
        else:
            value = type(getattr(opt, k))(v)
    except Exception as e:
        raise TypeError(e)

    return value


def format_key_value(opt, k, v, **kwargs):
    value = parse_types(opt, k, v)
    if k == 'batch_size':
        value = max(1, value)  # '16'
    if k == 'project':
        value = f'{kwargs["data_path"]}/{v[1:] if v.startswith("/") else v}'
    if k == 'workers':
        value = max(0, value)
    if k == 'epochs':
        value = max(5, value)
    if k == 'nc':
        value = len(getattr(opt, 'names')) if value == 0 else v
    if k == 'names':
        if len(value) == 0:
            raise ValueError('Class names should be equal to nc!')

    if k in ['train', 'val'] and value == '':
        value = os.path.join(kwargs["data_path"], f'{k}_dataset.pkl')
    return value


def check_opts(opt, custom_cfg=None, data_path='data'):
    unmatched_configs = custom_cfg

    if custom_cfg:
        configs = get_configs(custom_cfg)
        unmatched_configs = {}

        kwargs = {'data_path': data_path}
        for k, v in configs.items():
            k = k.replace('-', '_')
            if hasattr(opt, k):
                value = format_key_value(opt, k, v, **kwargs)
                setattr(opt, k, value)
            else:
                unmatched_configs[k] = v
    return opt, unmatched_configs


def main():
    pass


if __name__ == '__main__':
    main()
