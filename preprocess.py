import numpy as np


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_images(image_a, image_b, type_a, type_b):
    """根据输入的影像类型进行预处理。

    支持的组合：
      - opt/opt：min-max 归一化到 [0,1]
      - sar/opt 或 opt/sar：按类型分别预处理（SAR：log(1+x)+标准化；OPT：标准化）
      - sar/sar：两张 SAR 都做 log(1+x)+标准化
    """
    if type_a == 'opt' and type_b == 'opt':
        # 保持原行为：双光学采用 min-max
        image_a = normalize_img(image_a)
        image_b = normalize_img(image_b)
    elif (type_a in ['sar', 'opt']) and (type_b in ['sar', 'opt']):
        # 混合模态或双 SAR：按类型分别处理
        image_a = preprocess_img(image_a, type_a)
        image_b = preprocess_img(image_b, type_b)
    else:
        raise ValueError(f"Unknown image type pair: {type_a}/{type_b}")
    return image_a, image_b


def normalize_img(image):
    min_val = torch.min(image)
    max_val = torch.max(image)
    if max_val - min_val != 0:
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        small_constant = 1e-10
        normalized_image = (image - min_val) / (max_val - min_val + small_constant)
    return normalized_image
def preprocess_img(data, d_type):
    pps_data = data.detach().to(device).float()

    if d_type == 'opt':
        pps_data = stad_img(pps_data, channel_first=False)
    elif d_type == 'sar':
        pps_data[torch.abs(pps_data) <= 0] = torch.min(pps_data[torch.abs(pps_data) > 0]).float()
        pps_data = torch.log(pps_data + 1.0).float()
        pps_data = stad_img(pps_data, channel_first=False)
    return pps_data.float()


def stad_img(img, channel_first):
    img = img.float()  # Make sure the input image is of type float
    if channel_first:
        channel, img_height, img_width = img.shape
        img = img.reshape(channel, -1).float()
        mean = torch.mean(img, dim=1, keepdim=True).float()
        center = (img - mean).float()
        var = torch.sum(center**2, dim=1, keepdim=True).float() / (img_height * img_width)
        std = torch.sqrt(var).float()
        nm_img = (center / std).float()
        nm_img = nm_img.reshape(channel, img_height, img_width).float()
    else:
        img_height, img_width, channel = img.shape
        img = img.reshape(-1, channel).float()
        mean = torch.mean(img, axis=0, keepdim=True).float()
        center = (img - mean).float()
        var = torch.sum(center**2, axis=0, keepdim=True).float() / (img_height * img_width)
        std = torch.sqrt(var).float()
        nm_img = (center / std).float()
        nm_img = nm_img.reshape(img_height, img_width, channel).float()
    return nm_img.float()



