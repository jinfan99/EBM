from pathlib import Path

import random
from PIL import Image

import torch as th
from torchvision import transforms as T
from ebm_finetune.train_util import pil_image_to_norm_tensor


import numpy as np
from torch.utils import data
import os.path as osp
from glob import glob
import random 

def random_resized_crop(image, shape, resize_ratio=1.0):
    """
    Randomly resize and crop an image to a given size.

    Args:
        image (PIL.Image): The image to be resized and cropped.
        shape (tuple): The desired output shape.
        resize_ratio (float): The ratio to resize the image.
    """
    image_transform = T.RandomResizedCrop(shape, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0))
    return image_transform(image)



def get_shared_stems(image_files_dict, text_files_dict):
    image_files_stems = set(image_files_dict.keys())
    text_files_stems = set(text_files_dict.keys())
    return list(image_files_stems & text_files_stems)


class blender_64(data.Dataset):
    def __init__(self, 
                 data_dir="./scratch/EBM/cgm_data/",
                 uncond_p=0.05,
                ):


        data =np.load(data_dir+"/clevr_training_data_128.npz",allow_pickle=True)
        # self.ims_1 = data["arr_0"]
        # self.labels_1 = np.array(data["arr_1"])
        self.ims_1 = data['ims']
        self.labels_1 = data['labels']

        print(f"size of the data : {self.ims_1.shape}, {self.labels_1.shape}")
            
        # self.ims_1= self.ims_1[shard:][::num_shards]
        # self.labels_1 = self.labels_1[shard:][::num_shards]
      

        self.label_description = {
            "left": "to the left of",
            "right": "to the right of",
            "behind": "behind",
            "front": "in front of",
            "above": "above",
            "below": "below"
            }


        self.colors_to_idx = {"gray": 0, "red": 1, "blue": 2, "green": 3, "brown": 4, "purple": 5, "cyan": 6, "yellow": 7, "none": 8}
        # self.shapes_to_idx = {"cube": 0, "boot": 1, "sphere": 2,"truck":3,"cylinder":4,"none": 5}
        self.shapes_to_idx = {"cube": 0, "sphere": 1,"cylinder":2,"none": 3}
        self.materials_to_idx = {"rubber": 0, "metal": 1, "none": 2}
        self.sizes_to_idx = {"small": 0, "large": 1,"none":2}
        self.relations_to_idx = {"left": 0, "right": 1, "front": 2, "behind": 3, "none": 4}
        

        self.colors = list(self.colors_to_idx.keys())
        self.shapes = list(self.shapes_to_idx.keys())
        self.materials = list(self.materials_to_idx.keys())
        self.sizes = list(self.sizes_to_idx.keys())
        self.relations = list(self.relations_to_idx.keys())

        self.uncond_p = uncond_p # 0.05
        self.size = self.labels_1.shape[0]


        print('image data size', self.ims_1.shape)
        print('label data size', self.labels_1.shape)


    def __len__(self):
        return self.size

    def __getitem__(self, index):
    
        im_1 = Image.fromarray(self.ims_1[index]).resize((64, 64))
        label_1 = self.labels_1[index]
     
       
        mask = random.random() > self.uncond_p

        base_tensor = pil_image_to_norm_tensor(im_1)
   
        return  th.tensor(label_1,dtype=th.long),th.tensor(mask, dtype=th.bool), base_tensor

    def get_test_sample(self):

        label = [2] # Cylinder 

        description = self._convert_caption(label).strip()
        print(f"The label:{label} corresponding to the caption: {description}")

        return {"caption":description,"label":th.tensor(label,dtype=th.long)}

  
    def _convert_caption(self, label):


        return f'A {self.shapes[label[0]]}'
    
class shapenet_128(data.Dataset):
    def __init__(self, 
                 data_dir="./scratch/EBM/srn_cars/",
                 train_only=True,
                 uncond_p=0.05
                ):

        self.img_path_list = []
        train_sample_list = glob(osp.join(data_dir, 'cars_train', '*'))
        print('loading images from training set...')
        for sample_dir in train_sample_list:
            all_img = glob(osp.join(sample_dir, 'rgb', '*.png'))
            self.img_path_list += all_img
        print('images from training set loaded!')
        
        if not train_only:  # get the directory list for all of the objects
            val_sample_list = glob(osp.join(data_dir, 'cars_val', '*'))
            test_sample_list = glob(osp.join(data_dir, 'cars_test', '*'))
            
            print('loading images from val and test set...')
            for sample_list in [val_sample_list, test_sample_list]:
                for sample_dir in sample_list:
                    all_img = glob(osp.join(sample_dir, 'rgb', '*.png'))
                    selected_img = random.sample(all_img, 50)   # only sample 50 samples, same as training set 
                    self.img_path_list += selected_img
            print('images from training set loaded!')
        else:
            print('only using images from training set')

        self.uncond_p = uncond_p # 0.05


    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = Image.open(self.img_path_list[index]).convert('RGB')
        # img = img.resize((64, 64))
        label = 0   # always set label to be 0
       
        mask = random.random() > self.uncond_p

        base_tensor = pil_image_to_norm_tensor(img)
   
        return  th.tensor(label,dtype=th.long),th.tensor(mask, dtype=th.bool), base_tensor

    def get_test_sample(self):

        return None
      


# if __name__=="__main__":

    
#     dataset = blender_64()
#     print(dataset.__getitem__(0)[0])