import torch


def dice_func(output, target):
    smooth = 0.001
    output = torch.as_tensor(output,dtype=torch.float32)
    target = torch.as_tensor(target,dtype=torch.float32)
    intersection = output*target
    intersection_sum = torch.sum(intersection)
    # print('inter=',intersection_sum)
    output_sum = output.sum()
    # print('output=',output_sum)
    target_sum = target.sum()
    # print('target=',target_sum)
    f = 2*intersection_sum / (output_sum + target_sum +smooth)

    return f
