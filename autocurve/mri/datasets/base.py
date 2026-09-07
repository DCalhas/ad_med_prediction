import time

import h5py
import torch
import os
import nibabel
import numpy as np
import glob

from torchvision.transforms import v2

class DatasetMRI(torch.utils.data.Dataset):

    def __init__(self, patient_ids=None, 
                 dataset_dir=None, h5dataset_path=None, h5dataset_name=None, glob_regx="*/*/*/*.BET.FLIRT.nii.gz",
                transform_output=False, transform_output_size=(100, 100), force=False):

        super(DatasetMRI, self).__init__()
        
        if h5dataset_path is not None:
            if h5dataset_name is None:
                raise ValueError("h5dataset_name must be provided if h5dataset_path is provided.")
        
        self.h5dataset_path = h5dataset_path
        self.h5dataset_name = h5dataset_name
        
        self.transforms = v2.Compose([
                v2.RandomRotation((-90,90)),
                v2.RandomResizedCrop(transform_output_size, ratio=(0.4, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.Normalize(mean=[0.4419], std=[0.2276])
        ])
        self.transform_output = transform_output
        
        print("I: Searching for MRI files in dataset directory...", end="\n")
        self.mri_paths = glob.glob(f'{dataset_dir}/{glob_regx}')
        self.N = len(self.mri_paths)
        self.resolution = nibabel.load(self.mri_paths[0]).get_fdata().shape if self.N > 0 else (0, 0, 0)
        
        self.mri_paths = np.sort(self.mri_paths)
        
        self.data_pointer = None
        if os.path.isfile(f"{self.h5dataset_path}/{self.h5dataset_name}.h5") and not force:
            print("I: h5 dataset file already exists. Loading h5 file toggle force to skip.", end="\n")
            data_pointer = h5py.File(f"{self.h5dataset_path}/{self.h5dataset_name}.h5", "r")
            self.N = data_pointer["adni"].shape[0]
            self.resolution = data_pointer["adni"].shape[1:]
            data_pointer.close()
        
        self.patients = np.arange(self.N)
        
        if patient_ids is not None:
            self.patients = []
            for i, path in enumerate(self.mri_paths):
                path_parts = path.split("/")
                if path_parts[7] in patient_ids:
                    self.patients.append(i)
            self.patients = np.array(list(self.patients))
        
        self.N = self.patients.shape[0]
        
        # self.cache = [None] * self.N 
        
    def __len__(self,): 
        #how many mris are there
        return self.N
        

    def getitem_path(self, idx):
        idx = self.patients[idx]
        mri_path = self.mri_paths[idx] #load mri of the corresponding index
        instance = nibabel.load(mri_path).get_fdata()
        return instance.astype(np.float32)
    
    def get_patient_ids(self):
        return list({self.mri_paths[i].split("/")[-4] for i in self.patients})

    def __getitem__(self, idx):
        idx = self.patients[idx]
        # return self.__getitem_path(idx)
        # if self.cache[idx] is not None:
            # image = self.cache[idx]
            # return self._transfrom_image(image) if self.transform_output else image
        if self.data_pointer is None:
            self.data_pointer=h5py.File(f"{self.h5dataset_path}/{self.h5dataset_name}.h5", "r")
        image = self.data_pointer["adni"][idx]
        # self.cache[idx] = image
        return self._transfrom_image(image) if self.transform_output else image
        
    def _transfrom_image(self, image):
        selected = np.linspace(50, 125, 20).astype(int) # TODO: add this as an argument to the dataset class
        image = image[:,:,selected]
        mri = torch.tensor(image.copy(), dtype=torch.float32).permute(2, 0, 1)
        mri = self.transforms(mri)
        
        image = mri.permute(1, 2, 0).numpy()
        
        return image

    def set_data_pointer(self, force=False):
        if self.data_pointer is not None:
            print("I: Data pointers already set.", end="\n")
            return
        if(os.path.isfile(f"{self.h5dataset_path}/{self.h5dataset_name}.h5")):
            if not force:
                print("I: Data pointers h5 file already exists. Toggle force to create a new one.", end="\n")
                return
            print("I: Deleting existing h5 file.", end="\n")
            os.remove(f"{self.h5dataset_path}/{self.h5dataset_name}.h5")
                
        file_pointer=h5py.File(f"{self.h5dataset_path}/{self.h5dataset_name}.h5", "w")
        file_pointer.create_dataset("adni", (len(self.mri_paths), *self.resolution), chunks=(1, *self.resolution), dtype=np.float32, compression='gzip')
        for idx in range(len(self.mri_paths)):
            start_time=time.time()
            mri_path = self.mri_paths[idx] #load mri of the corresponding index
            instance = nibabel.load(mri_path).get_fdata() 
            instance = instance.astype(np.float32) # TODO: supres warnings about precision loss when converting to float32
            file_pointer["adni"][idx]=instance
            print("I: Setting data pointer "+str(idx+1)+"/"+str(len(self.mri_paths))+" took "+str(time.time()-start_time)+" seconds", end="\r")

        file_pointer.close()
        self.data_pointer = h5py.File(f"{self.h5dataset_path}/{self.h5dataset_name}.h5", "r")
        print("I: Successfuly set data pointers!", end="\n")