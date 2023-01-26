# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import torch

from torchvision import transforms
from torchvision.transforms import functional as F


class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F._get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w

def rand_crop_flow(image,label, flows_forward, flows_backward, crop_size):
    _, t, _, _, _ =label.size()
    new_t = random.randint(0, t - crop_size)


    image = image[:,new_t: new_t + crop_size,:,:,:]
    label = label[:, new_t: new_t + crop_size, :, :, :]
    flows_forward = flows_forward[:, new_t: new_t + crop_size - 1, :, :, :]
    flows_backward = flows_backward[:, new_t: new_t + crop_size-1, :, :, :]
    return image,label,flows_forward,flows_backward

def rand_crop_onlycnn(image,label,crop_size):
    _, t, _, _, _ =label.size()
    new_t = random.randint(0, t - crop_size)

    image = image[:,new_t: new_t + crop_size,:,:,:]
    label = label[:, new_t: new_t + crop_size, :, :, :]
    return image,label