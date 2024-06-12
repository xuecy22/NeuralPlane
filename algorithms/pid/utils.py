import os
import yaml
import torch


def parse_config(filename):
    filepath = os.path.join(get_root_dir(), 'config', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('ControlConfig', (object,), config_data)

def get_root_dir():
    return os.path.split(os.path.realpath(__file__))[0]

def get_diff_angle(loc1, loc2):
    diff_vector = loc2 - loc1
    diff_y = diff_vector[:, 1].reshape(-1, 1)
    diff_x = diff_vector[:, 0].reshape(-1, 1)
    return torch.atan2(diff_y, diff_x)

def get_length(vector):
    return torch.sqrt(torch.sum(vector * vector, dim=1, keepdim=True))

def get_vector_dot(vec1, vec2):
    return torch.sum(vec1 * vec2, dim=1, keepdim=True)

def get_cross_error(vec1, vec2):
    x1 = vec1[:, 0].reshape(-1, 1)
    y1 = vec1[:, 1].reshape(-1, 1)
    x2 = vec2[:, 0].reshape(-1, 1)
    y2 = vec2[:, 1].reshape(-1, 1)
    return x1 * y2 - y1 * x2

# 限制角度范围为[-pi, pi]
def wrap_PI(angle):
    res = wrap_2PI(angle)
    mask1 = res > torch.pi
    res -= 2 * torch.pi * mask1
    return res

def wrap_2PI(angle):
    res = angle % (2 * torch.pi)
    mask1 = res < 0
    res += 2 * torch.pi * mask1
    return res
