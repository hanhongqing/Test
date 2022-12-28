import numpy as np
import torch
from skimage.filters import threshold_otsu, rank
import tifffile
import math

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, Image): 
        self.Image = Image
        self.n_image = self.Image.shape[1]
        self.n_sample = args.n_sample
        self.size_image = args.size_image
        self.size_patch = args.size_patch
        self.intensity_max_DNA = args.intensity_max_DNA
        self.intensity_max_actin = args.intensity_max_actin
        self.patches = np.zeros([self.n_image * self.n_sample,2,self.size_patch,self.size_patch],dtype = np.float32)
        self.img_ids = np.zeros([self.n_image * self.n_sample],dtype = np.int64)
        for i in range(self.n_image):
            I_DNA = self.Image[0,i]
            I_pH3 = self.Image[1,i]
            I_actin = self.Image[2,i]
            thres = threshold_otsu(I_DNA)
            seg = I_DNA >= thres
            avg_pH3 = np.mean(I_pH3)
            for j in range(self.n_sample):
                while True:
                    x = np.random.randint(0,self.size_image - self.size_patch)
                    y = np.random.randint(0,self.size_image - self.size_patch)
                    patch_seg = seg[y:y+self.size_patch,x:x+self.size_patch]
                    patch_pH3 = I_pH3[y:y+self.size_patch,x:x+self.size_patch]
                    if np.sum(patch_seg) / self.size_patch**2 < args.thres_patch_fg:
                        continue
                    if np.mean(patch_pH3) > args.thres_patch_mito * avg_pH3:
                        continue
                    break
                self.patches[i*self.n_sample+j,0,:,:] =   I_DNA[y:y+self.size_patch,x:x+self.size_patch] / self.intensity_max_DNA
                self.patches[i*self.n_sample+j,1,:,:] = I_actin[y:y+self.size_patch,x:x+self.size_patch] / self.intensity_max_actin
                self.img_ids[i*self.n_sample+j    ] = i
                #tifffile.imwrite('temp/patch'+str(i)+str(j)+".tif",I_actin[y:y+self.size_patch,x:x+self.size_patch])
        
        
    
    def __len__(self):
        return self.n_image * self.n_sample


    def __getitem__(self,i):
        patch_np = self.patches[i]
        if np.random.random() < 0.5:
            patch_np = np.rot90(patch_np,k=1, axes=(1, 2))
        if np.random.random() < 0.5:
            patch_np = np.flip(patch_np,axis=1)
        if np.random.random() < 0.5:
            patch_np = np.flip(patch_np,axis=2)
        patch_torch = torch.from_numpy(patch_np.copy())
        patch_torch[0] = (patch_torch[0] - torch.mean(patch_torch[0])) / (max(torch.std(patch_torch[0]), 1.0 / self.size_patch))
        patch_torch[1] = (patch_torch[1] - torch.mean(patch_torch[1])) / (max(torch.std(patch_torch[1]), 1.0 / self.size_patch))
        return patch_torch,self.img_ids[i]



