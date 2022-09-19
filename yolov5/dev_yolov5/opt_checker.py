import argparse
import os
import yaml
import distutils


def check_opts(opt, custom_cfg=None, data_path='data'):
    unmatched_configs = custom_cfg
    if custom_cfg:
        convert_back = False
        if isinstance(custom_cfg, str):
            with open(custom_cfg, 'r') as stream:
                configs = yaml.safe_load(stream)
        elif isinstance(custom_cfg, dict):
            configs = custom_cfg
        unmatched_configs = {}
        if isinstance(opt, dict):
            opt = argparse.Namespace(**opt)
            convert_back = True
        for k, v in configs.items():
            k = k.replace('-', '_')
            if hasattr(opt, k):
                _type = type(getattr(opt, k))

                if k == 'project':
                    if v.startswith('/'):
                        v = v[1:]
                try:
                    if _type == list:
                        content_type = None
                        if len(getattr(opt, k)) > 0:
                            content_type = type(getattr(opt, k)[0])
                        if isinstance(v, list):
                            value = v
                        else:
                            value = [
                                content_type(key) if content_type is not None else key for key in
                                v.strip('][').split(', ') if key != ''
                            ]
                    elif _type == type(None):
                        value = v
                    elif _type == bool:
                        value = bool(distutils.util.strtobool(v))
                    else:
                        value = type(getattr(opt, k))(v)

                    # if k == 'anchors':
                    #     value = '3.0'
                    if k == 'batch_size':
                        value = max(1, value)  # '16'
                    # if k == 'device':
                    #     value = 'cpu'
                    # if k == 'exist_ok':
                    #     value = 'false'
                    # if k == 'name':
                    #     value = ''
                    # if k == 'noautoanchor':
                    #     value = 'false'
                    # if k == 'noplots':
                    #     value = 'false'
                    # if k == 'nosave':
                    #     value = 'false'
                    # if k == 'noval':
                    #     value = 'false'
                    # if k == 'optimizer':
                    #     value = 'SGD'
                    # if k == 'path':
                    #     value = ''
                    if k == 'project':
                        value = f'{data_path}/{v}'
                    if k == 'workers':
                        value = max(0, value)
                    if k == 'epochs':
                        value = max(5, value)
                    if k == 'nc':
                        value = len(getattr(opt, 'names')) if value == 0 else v
                    if k == 'names':
                        if len(value) == 0:
                            raise ValueError('Class names should be equal to nc!')
                    if k == 'train' and value == '':
                        value = os.path.join(data_path, 'train_dataset.pkl')
                    if k == 'val' and value == '':
                        value = os.path.join(data_path, 'val_dataset.pkl')
                    if k == 'test' and value == '':
                        value = os.path.join(data_path, 'test_dataset.pkl')
                    setattr(opt, k, value)
                except:
                    raise TypeError('Type mismatch')
            else:
                unmatched_configs[k] = v
        if convert_back:
            opt = vars(opt)
    return opt, unmatched_configs


def main():
    pass


if __name__ == '__main__':
    main()
