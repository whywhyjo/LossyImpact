import torch
import os 
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glymur
import cv2
    
class MammoDataset_concat (Dataset):
    def __init__(self,cases,root):
        self.root = root
        self.cases = cases
        self.len = len(cases)
    def __getitem__(self, index):
        X_batch =[]
        case = self.cases[index]
        path = self.root[case]
        
        # binary mission
        if int(case[0]) ==0:
            y_batch = 1 # malignant
        else:
            y_batch =0 # normal / benign
        #y_batch = int(case[0])        
        tmp = np.load(path+'/mammo.npz')['arr_0']
        if tmp.shape[1] == 1:
            tmp = np.squeeze(tmp,1)
        tmp = np.concatenate([np.concatenate([tmp[0],tmp[1]],axis=0),np.concatenate([tmp[2],tmp[3]],axis=0)],axis=1)
        tmp = tmp[int(tmp.shape[0]*0.2):int(tmp.shape[0]*0.8),int(tmp.shape[1]*0.3):int(tmp.shape[1]*0.7)]
        tmp = (tmp-tmp.min()) / (tmp.max()-tmp.min())
        X_batch.append(tmp)
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        if X_batch.shape[1] != 1:
            X_batch = np.reshape(X_batch,(1,tmp.shape[0],tmp.shape[1]))
        return torch.from_numpy(X_batch).float(), torch.from_numpy(y_batch).float()
    def __len__(self):
        return self.len


class MammoDataset (Dataset):
    def __init__(self,cases,root,percent):
        self.root = root 
        self.cases = cases
        self.len = len(cases)
        self.x_max = 4096
        self.y_max = 3328
        self.percent = percent
        if percent <1.0: 
            self.x_max = int(4096*percent)+1
            self.y_max = int(3328*percent)+1

    def __getitem__(self, index): 
        case = self.cases[index]
        path = self.root[case]        
        # binary mission
        if int(case[0]) ==0:
            y_batch = 1 # malignant
        else:
            y_batch =0 # normal / benign
        #y_batch = int(case[0])     
        y_batch = np.array(y_batch)

        ##### resizing and padding   
        img_rmlo = glymur.Jp2k(os.path.join(path,'RMLO.jp2'))[:]
        if self.percent <1.0:
            img_rmlo = cv2.resize(img_rmlo,None, fx=self.percent, fy=self.percent, interpolation=cv2.INTER_AREA)
        img_rmlo = np.pad(img_rmlo,((self.x_max-img_rmlo.shape[0], 0), (self.y_max-img_rmlo.shape[1], 0)),mode='constant')

        img_rcc = glymur.Jp2k(os.path.join(path,'RCC.jp2'))[:]
        if self.percent <1.0:
            img_rcc = cv2.resize(img_rcc,None, fx=self.percent, fy=self.percent, interpolation=cv2.INTER_AREA)
        img_rcc = np.pad(img_rcc,((0,self.x_max-img_rcc.shape[0]), (self.y_max-img_rcc.shape[1], 0)),mode='constant')

        img_lmlo = glymur.Jp2k(os.path.join(path,'LMLO.jp2'))[:]
        if self.percent <1.0:
            img_lmlo = cv2.resize(img_lmlo,None, fx=self.percent, fy=self.percent, interpolation=cv2.INTER_AREA)
        img_lmlo = np.pad(img_lmlo,((self.x_max-img_lmlo.shape[0], 0), (0,self.y_max-img_lmlo.shape[1])),mode='constant')


        img_lcc = glymur.Jp2k(os.path.join(path,'LCC.jp2'))[:]
        if self.percent <1.0:
            img_lcc = cv2.resize(img_lcc,None, fx=self.percent, fy=self.percent, interpolation=cv2.INTER_AREA)
        img_lcc = np.pad(img_lcc,((0,self.x_max-img_lcc.shape[0]), (0,self.y_max-img_lcc.shape[1])),mode='constant')


        tmp = np.concatenate([np.concatenate([img_rmlo,img_rcc],axis=0),np.concatenate([img_lmlo,img_lcc],axis=0)],axis=1)
        tmp = tmp[int(tmp.shape[0]*0.2):int(tmp.shape[0]*0.8),int(tmp.shape[1]*0.3):int(tmp.shape[1]*0.7)]
        tmp = (tmp-tmp.min()) / (tmp.max()-tmp.min())

        X_batch = np.reshape(tmp,(1,tmp.shape[0],tmp.shape[1]))

        return torch.from_numpy(X_batch).float(), torch.from_numpy(y_batch).float()

    def __len__(self):
        return self.len