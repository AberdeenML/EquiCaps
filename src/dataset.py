# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified by Athinoulla Konstantinou in 2025.

from torch.utils.data import Dataset
import torch
import torchvision
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R

class Dataset3DIEBench(Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience    

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        img_1 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[0]}.jpg")
        img_2 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[1]}.jpg")         
    
        angles_1 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[0]}.npy")[:3].astype(np.float32)
        angles_2 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[1]}.npy")[:3].astype(np.float32)
        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        rot_1_to_2 = rot_1.inv()*rot_2
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)

        return img_1, img_2, torch.FloatTensor(angles), label

    def __len__(self):
        return len(self.samples)

class Dataset3DIEBench_and_3DIEBenchT_matrix(Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):

        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        img_1 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[0]}.jpg")
        img_2 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[1]}.jpg")         
    
        angles_1 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[0]}.npy")[:3].astype(np.float32)
        angles_2 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[1]}.npy")[:3].astype(np.float32)
        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        rot_1_to_2 = rot_1.inv()*rot_2
        matrix = rot_1_to_2.as_matrix().astype(np.float32)
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)

        return img_1, img_2, torch.FloatTensor(angles), torch.FloatTensor(matrix), label

    def __len__(self):
        return len(self.samples)

class Dataset3DIEBench_T_RotTranslation(Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience    

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        img_1 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[0]}.jpg")
        img_2 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[1]}.jpg")         
    
        latent_1 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[0]}.npy").astype(np.float32)
        latent_2 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[1]}.npy").astype(np.float32)
        angles_1 = latent_1[:3]
        angles_2 = latent_2[:3]
        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        rot_1_to_2 = rot_1.inv()*rot_2
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)
        
        # extract the object frame translation parameters
        translation = latent_2[-3:] - latent_1[-3:]
        latent_total = np.concatenate((angles,translation))

        return img_1, img_2, torch.FloatTensor(latent_total), label

    def __len__(self):
        return len(self.samples)

class Dataset3DIEBench_T_RotTranslationCol(Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience    

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        img_1 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[0]}.jpg")
        img_2 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[1]}.jpg")         
    
        latent_1 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[0]}.npy").astype(np.float32)
        latent_2 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[1]}.npy").astype(np.float32)
        angles_1 = latent_1[:3]
        angles_2 = latent_2[:3]
        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        rot_1_to_2 = rot_1.inv()*rot_2
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)
        
        # extract the object frame translation parameters
        translation = latent_2[-3:] - latent_1[-3:]
        # extract the color parameters
        color_params = latent_2[[3,6]] - latent_1[[3,6]] 

        latent_total = np.concatenate((angles,translation,color_params))

        return img_1, img_2, torch.FloatTensor(latent_total), label

    def __len__(self):
        return len(self.samples)

def get_object_frame_matrix(angles, translation):
    rotation_matrix = R.from_euler('xyz', angles).as_matrix()

    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = rotation_matrix

    T_4x4 = np.eye(4)
    T_4x4[:3, 3] = translation

    M = T_4x4 @ R_4x4  
    return M

def get_base_frame_matrix(angles, translation):
    rotation_matrix = R.from_euler('xyz', angles).as_matrix()

    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = rotation_matrix

    T_4x4 = np.eye(4)
    T_4x4[:3, 3] = translation

    M = R_4x4 @ T_4x4
    return M

class Dataset3DIEBenchT_matrix_object_frame(Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience    

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        img_1 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[0]}.jpg")
        img_2 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[1]}.jpg")         
    
        latent_1 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[0]}.npy").astype(np.float32)
        latent_2 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[1]}.npy").astype(np.float32)
       
        angles_1 = latent_1[:3]
        angles_2 = latent_2[:3]
        
        translation_1 = latent_1[-3:]
        translation_2 = latent_2[-3:]
       
        M1 = get_object_frame_matrix(angles_1, translation_1)
        M2 = get_object_frame_matrix(angles_2, translation_2) 
        
        M12 = np.linalg.inv(M1) @ M2
         
        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        rot_1_to_2 = rot_1.inv()*rot_2

        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)

        return img_1, img_2, torch.FloatTensor(angles), torch.FloatTensor(M12), label

    def __len__(self):
        return len(self.samples)
    
class Dataset3DIEBenchT_matrix_base_frame(Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience    

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        img_1 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[0]}.jpg")
        img_2 = self.get_img(self.dataset_root / self.samples[i][1:] / f"image_{views[1]}.jpg")         
    
        latent_1 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[0]}.npy").astype(np.float32)
        latent_2 =np.load(self.dataset_root / self.samples[i][1:] / f"latent_{views[1]}.npy").astype(np.float32)
        
        angles_1 = latent_1[:3]
        angles_2 = latent_2[:3]
        
        translation_1 = latent_1[-3:]
        translation_2 = latent_2[-3:]
        
        M1 = get_base_frame_matrix(angles_1, translation_1)
        M2 = get_base_frame_matrix(angles_2, translation_2) 

        M12 = np.linalg.inv(M1) @ M2

        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        
        rot_1_to_2 = rot_1.inv()*rot_2
        
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)

        return img_1, img_2, torch.FloatTensor(angles), torch.FloatTensor(M12), label

    def __len__(self):
        return len(self.samples)