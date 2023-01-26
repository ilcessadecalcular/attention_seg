import os
import SimpleITK as sitk
import torch
import random
import numpy as np
import cv2

class MedData_train_onlycnn(torch.utils.data.Dataset):
    def __init__(self, source_dir, label_dir, crop_size):
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.patient_dir = os.listdir(label_dir)
        self.crop_size = crop_size
        # self.crop_size = crop_size

    def __len__(self):
        return len(self.patient_dir)

    def znorm(self, image):
        mn = image.mean()
        sd = image.std()
        return (image - mn) / sd

    def rand_crop_onlycnn(self,image, label, crop_size):
        t, _, _, _ = label.size()
        new_t = random.randint(0, t - crop_size)

        image = image[new_t: new_t + crop_size, :, :, :]
        label = label[new_t: new_t + crop_size, :, :, :]
        return image, label

    def __getitem__(self, index):
        patient = self.patient_dir[index]
        source_path = os.path.join(self.source_dir, patient)
        label_path = os.path.join(self.label_dir,patient)
        image_source = sitk.ReadImage(source_path)
        source_array = sitk.GetArrayFromImage(image_source)
        image_label = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(image_label)
        znorm_source_array = self.znorm(source_array)
        # crop_source_array, crop_label_array = self.rand_crop(znorm_source_arrary,label_array)
        out_source_array = znorm_source_array[:, np.newaxis]
        out_source_array = torch.FloatTensor(out_source_array)
        out_label_array = label_array[:, np.newaxis]
        out_label_array = torch.FloatTensor(out_label_array)
        out_source_array, out_label_array = self.rand_crop_onlycnn(out_source_array,out_label_array,self.crop_size)
        return out_source_array,out_label_array

class MedData_val_onlycnn(torch.utils.data.Dataset):
    def __init__(self, source_dir, label_dir):
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.patient_dir = os.listdir(label_dir)


    def __len__(self):
        return len(self.patient_dir)

    def znorm(self, image):
        mn = image.mean()
        sd = image.std()
        return (image - mn) / sd

    def __getitem__(self, index):
        patient = self.patient_dir[index]
        source_path = os.path.join(self.source_dir, patient)
        label_path = os.path.join(self.label_dir,patient)
        image_source = sitk.ReadImage(source_path)
        source_array = sitk.GetArrayFromImage(image_source)
        image_label = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(image_label)
        znorm_source_array = self.znorm(source_array)
        out_source_array = znorm_source_array[:, np.newaxis]
        out_source_array = torch.FloatTensor(out_source_array)
        out_label_array = label_array[:, np.newaxis]
        out_label_array = torch.FloatTensor(out_label_array)
        return out_source_array,out_label_array


class MedData_train_flow(torch.utils.data.Dataset):
    def __init__(self, source_dir, label_dir):
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.patient_dir = os.listdir(label_dir)
        # self.crop_size = crop_size

    def __len__(self):
        return len(self.patient_dir)

    def normalize_hu(self,image):
        # 将输入图像的像素值(200 ~ 600)归一化到0~1之间
        MIN_BOUND = 200
        MAX_BOUND = 500
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return (image * 255).astype(np.uint8)

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        t, h, w = lrs.shape
        lrs_input = self.normalize_hu(lrs)

        dis = cv2.DISOpticalFlow_create(2)  # PRESET_ULTRAFAST, PRESET_FAST and PRESET_MEDIUM

        flows_forward = []
        flows_backward = []
        for i in range(t):
            if i == 0:
                flow_backward = None
            else:
                flow_backward = dis.calc(lrs_input[i],lrs_input[i-1], None)
                flow_backward = torch.FloatTensor(flow_backward)
                flows_backward.append(flow_backward)
            if i == t-1:
                flow_forward = None
            else:
                flow_forward = dis.calc(lrs_input[i], lrs_input[i+1], None)
                flow_forward = torch.FloatTensor(flow_forward)
                flows_forward.append(flow_forward)

        return torch.stack(flows_forward, dim=0), torch.stack(flows_backward, dim=0)

    def znorm(self, image):
        mn = image.mean()
        sd = image.std()
        return (image - mn) / sd

    def __getitem__(self, index):
        patient = self.patient_dir[index]
        source_path = os.path.join(self.source_dir, patient)
        label_path = os.path.join(self.label_dir,patient)
        image_source = sitk.ReadImage(source_path)
        source_array = sitk.GetArrayFromImage(image_source)
        image_label = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(image_label)
        flows_forward , flows_backward = self.compute_flow(source_array)
        znorm_source_array = self.znorm(source_array)
        out_source_array = znorm_source_array[:, np.newaxis]
        out_source_array = torch.FloatTensor(out_source_array)
        out_label_array = label_array[:, np.newaxis]
        out_label_array = torch.FloatTensor(out_label_array)
        return out_source_array,out_label_array,flows_forward,flows_backward


def z_norm(image):
    mn = image.mean()
    sd = image.std()
    return (image - mn) / sd

def normalize_hu(image):
    # 将输入图像的像素值(200 ~ 600)归一化到0~1之间
    image = image.numpy()
    MIN_BOUND = 200
    MAX_BOUND = 600
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return (image*255).astype(np.uint8)


if __name__ == "__main__":
    source_dir = 'test/source'
    label_dir = 'test/label'
    dataset_train = MedData_train(source_dir,label_dir)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=2)
    for epoch in range(10):
        for ite, sample in enumerate(train_loader):
            img = sample[0]
            label = sample[1]
            # label = target
            print(epoch, ite)
            print(img.size())
            print(label.size())