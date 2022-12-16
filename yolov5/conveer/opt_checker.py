import os
import distutils
import argparse


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

        # get original opt dtype
        if len(getattr(opt, k)) > 0:
            content_type = type(getattr(opt, k)[0])

        # format data
        data = (json.loads(v) if isinstance(v, str) else v)
        if not isinstance(data, list):
            data = [data, data]
        value = [content_type(val) for val in data]
    except:
        print('problems')
    return value


def parse_types(opt, k, v):
    _type = type(getattr(opt, k))
    if v is None:
        print(f'ERRO: value from customer config with key {k} equals "None". Use default value "{getattr(opt, k)}"')
        return getattr(opt, k)
    try:
        if _type == list:
            value = parse_list(opt, k, v)
        elif _type is type(None):
            value = v
        elif _type == bool:
            value = bool(distutils.util.strtobool(v)) if type(v) != bool else v
        else:
            value = type(getattr(opt, k))(v)
    except:
        raise TypeError('Type mismatch')

    return value


def format_key_value(opt, configs, k, v):
    value = parse_types(opt, k, v)
    if k == 'batch_size':
        value = max(1, value)  # '16'
    if k == 'project':
        # value = f'{kwargs["data_path"]}'  # /{v[1:] if v.startswith("/") else v}'
        value = f'{configs["path_to_data"]}'
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
        value = os.path.join(configs["path_to_data"], f'{k}_dataset.pkl')
    return value


def body_v1(opt, configs):
    convert_to_dict = False
    if not isinstance(opt, argparse.Namespace):
        opt = argparse.Namespace(**opt)
        convert_to_dict = True
    unmatched_configs = {}
    for k, v in configs.items():
        k = k.replace('-', '_')
        if hasattr(opt, k):
            value = format_key_value(opt, configs, k, v)
            if vars(opt)[k] != value:
                print(f'key = {k}\tcurrent_key = {vars(opt)[k]}\tnew_key = {value}')
                setattr(opt, k, value)
        else:
            unmatched_configs[k] = v
    if convert_to_dict:
        opt = vars(opt)
    return opt, unmatched_configs


def body_v2(opt, configs, **kwargs):
    unmatched_configs = {}
    opt_keys = set(vars(opt).keys() if not isinstance(opt, dict) else opt.keys())
    configs_keys = set(configs.keys())
    intersected_keys = list(opt_keys.intersection(configs_keys))
    for key in intersected_keys:
        k = key.replace('-', '_')
        if hasattr(opt, k):
            value = format_key_value(opt, k, configs[k], **kwargs)
            if vars(opt)[k] != value:
                print(f'key = {k}\t\tcurrent_key = {vars(opt)[k]}\t\tnew_key = {value}')
                setattr(opt, k, value)
        else:
            unmatched_configs[k] = configs[k]

    return opt, unmatched_configs


def check_opts(opt, custom_cfg=None, version=1, path_to_data=None):
    unmatched_configs = custom_cfg
    if custom_cfg:
        configs = get_configs(custom_cfg)
        if path_to_data is not None:
            configs['path_to_data'] = path_to_data
        body_v = body_v1 if version == 1 else body_v2
        opt, unmatched_configs = body_v(opt, configs)
    return opt, unmatched_configs


def main():
    pass


if __name__ == '__main__':
    main()
