import os
import torch
import pandas as pd
from skimage import transform
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
import random


from skimage import img_as_ubyte

from tqdm import tqdm
import time
import os
import argparse
from PIL import Image

class Metrics:
    
    def __init__(self, pred, target, build=None):
        self.pred = pred
        self.target = target
        self.build = build
        pass
    
    @property
    def RMSE(self):
        with torch.no_grad():
            pred = self.pred
            target = self.target
            if self.build is not None:
                true_mask = (self.build == 0)
                pred = pred[true_mask]
                target = target[true_mask]
            t = (((pred-target)**2).mean())**0.5
            return t.detach().cpu().item()
    
    @property
    def MSE(self):
        with torch.no_grad():
            pred = self.pred
            target = self.target
            if self.build is not None:
                true_mask = (self.build == 0)
                pred = pred[true_mask]
                target = target[true_mask]
            x = ((pred-target)**2).mean()
            return x.detach().cpu().item()
    
    @property
    def NMSE(self):
        with torch.no_grad():
            pred = self.pred
            target = self.target
            if self.build is not None:
                true_mask = (self.build == 0)
                pred = pred[true_mask]
                target = target[true_mask]
            t= ((pred-target)**2).mean()/(target**2).mean()
            return t.detach().cpu().item()


def RMSE(pred, target, metrics=None):
  loss = (((pred-target)**2).mean())**0.5
  return loss


def remove_outlier(preds, threshold_low=1/255):
    mask = preds.clone()
    mask[preds < threshold_low] = 0
    
    mask = mask.cuda()

    return mask


@torch.no_grad()
def eval_model(model, test_loader):

    # Set model to evaluate mode
    model.eval()

    n_samples = 0
    avg_loss = 0

    # check dataset type
    for inputs, targets, build_comp, _ , _ in tqdm(test_loader, desc="Evaluating the model.."):
        inputs = inputs.cuda()
        targets = targets.cuda()

        preds = model(inputs)  
        preds = torch.clip(preds, 0, 1)

        loss = RMSE(preds, targets) 

        avg_loss += (loss.item() * inputs.shape[0])
        n_samples += inputs.shape[0]

    avg_loss = avg_loss / (n_samples + 1e-7)

    return avg_loss

@torch.no_grad()
def inference_all_images(model, test_loader, output_dir: str):
    '''
    :output_dir: directory path where predicted images will be saved.
    '''
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print("Error: Failed to create the directory.")

    for inputs, _, _, building_names, Tx_names in tqdm(test_loader, desc="Saving estimated radio map images.."):
        inputs = inputs.cuda()
        preds = model(inputs)
        preds = torch.clip(preds, 0, 1)

        for i in range(len(preds)):
            pred = preds[i]
            pred = pred.reshape((256,256))

            # save predicted image
            io.imsave(os.path.join(output_dir, f'{building_names[i]}_{Tx_names[i]}.png'), img_as_ubyte(pred.cpu()))

    print('All predicted radio maps are saved.')
    

@torch.no_grad()
def inference_all_images_cal_loss(model, test_loader, output_dir: str, output_img=False):
    '''
    :output_dir: directory path where predicted images will be saved.
    '''
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print("Error: Failed to create the directory.")

    n_samples = 0
    avg_loss = 0
    for inputs, targets, _, building_names, Tx_names in tqdm(test_loader, desc="Saving estimated radio map images.."):
        inputs = inputs.cuda()
        preds = model(inputs)
        preds = torch.clip(preds, 0, 1)
        
        loss = RMSE(preds, targets.cuda()) 

        avg_loss += (loss.item() * inputs.shape[0])
        n_samples += inputs.shape[0]

        for i in range(len(preds)):
            pred = preds[i]
            pred = pred.reshape((256,256))
            target = targets[i].reshape((256,256))

            # save predicted image
            plt.imsave(os.path.join(output_dir, f'{building_names[i]}_{Tx_names[i]}.png'), pred.detach().cpu().numpy(), cmap='gray')
            plt.imsave(os.path.join(output_dir, f'{building_names[i]}_{Tx_names[i]}_gt.png'), target.detach().cpu().numpy(), cmap='gray')

    print('All predicted radio maps are saved.')
    avg_loss = avg_loss / (n_samples + 1e-7)
    print('Average loss: {:.8f}'.format(avg_loss))

    return avg_loss



class RadioMapTestset(Dataset):

    def __init__(self,
                 ind1=0,ind2=700, rnd_train=False, mixup_alpha=0.0, free_prog=False,
                 dataset_dir='./dataset',
                 input_dir="./png/",
                 gt_dir="./gain/",
                 numTx=80,                  
                 antenna = 'height',
                 cityMap="height",
                 transform= transforms.ToTensor()):

        
        self.ind1=ind1
        self.ind2=ind2
        
        self.numTx = numTx 
        
        # self.input_dir = input_dir
        # self.dir_gain = gt_dir
        
        self.input_dir = os.path.join(dataset_dir, input_dir)
        self.dir_gain = os.path.join(dataset_dir, gt_dir)
        
        # set buildings directory
        self.cityMap=cityMap
        if cityMap=="complete":
            self.dir_buildings=os.path.join(self.input_dir, "buildings_complete/")
            print("Using complete city map")
        elif cityMap=='height':
            self.dir_buildings = os.path.join(self.input_dir, "buildingsWHeight/")
        
        # set antenna directory
        self.antenna=antenna
        if antenna=='complete':
            self.dir_Tx = os.path.join(self.input_dir, "antennas/")
        elif antenna=='height':
            self.dir_Tx = os.path.join(self.input_dir, "antennasWHeight/")
        elif antenna=='building':
            self.dir_Tx = os.path.join(self.input_dir, "antennasBuildings/")

        self.transform= transform
        self.free_prog = free_prog
        if free_prog:
            self.dir_free_prog = os.path.join(self.input_dir, 'free_propagation')
        
        
        self.height = 256
        self.width = 256
        
        self.rnd_train = rnd_train
        self.mixup_alpha = mixup_alpha
        if self.mixup_alpha > 0.0:
            print("Mixup is enabled.")
            print('[Warning] This should be used only for training.')
            
        if rnd_train:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(2024)
            np.random.shuffle(self.maps)

        
    def __len__(self):
        return (self.ind2-self.ind1)*self.numTx
    
    # def __getitem__(self, idx):
        
    #     idxr=np.floor(idx/self.numTx).astype(int)
    #     idxc=idx-idxr*self.numTx 
    #     dataset_map_ind=idxr+self.ind1
    #     #names of files that depend only on the map:
    #     building_name = str(dataset_map_ind)
    #     Tx_name = str(idxc)
    #     name1 = building_name + ".png"
    #     #names of files that depend on the map and the Tx:
    #     name2 = building_name + "_" + Tx_name + ".png"
        
    #     #Load buildings:

    #     img_name_buildings = os.path.join(self.dir_buildings, name1)
    #     image_buildings = np.asarray(io.imread(img_name_buildings))   
        
    #     #Load Tx (transmitter):
    #     img_name_Tx = os.path.join(self.dir_Tx, name2)
    #     image_Tx = np.asarray(io.imread(img_name_Tx))
        
    #     #Load radio map:

    #     img_name_gain = os.path.join(self.dir_gain, name2)  
    #     image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
    #     # （256, 256, 1）, 0-1
        

    #     # Building complete for post processing
    #     self.post_buildings=os.path.join(self.input_dir, "buildings_complete/")
    #     build_comp_name = os.path.join(self.post_buildings, name1)
    #     build_comp = np.asarray(io.imread(build_comp_name))   

    #     inputs=np.stack([image_buildings, image_Tx], axis=2) 

    #     if self.transform:
    #         seed = random.randint(0, 2 ** 32)
    #         random.seed(seed)
    #         torch.manual_seed(seed)
    #         torch.cuda.manual_seed(seed)
    #         inputs = self.transform(inputs).type(torch.float32)
    #         random.seed(seed)
    #         torch.manual_seed(seed)
    #         torch.cuda.manual_seed(seed)
    #         image_gain = self.transform(image_gain).type(torch.float32)
    #         random.seed(seed)
    #         torch.manual_seed(seed)
    #         torch.cuda.manual_seed(seed)
    #         build_comp = self.transform(build_comp).type(torch.float32)
    #         #note that ToTensor moves the channel from the last asix to the first!
    #     return [inputs, image_gain, build_comp, building_name, Tx_name]
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=idxr+self.ind1
        #names of files that depend only on the map:
        if self.rnd_train:
            dataset_map_ind = self.maps[dataset_map_ind]
        building_name = str(dataset_map_ind)
        
        
        Tx_name = str(idxc)
        name1 = building_name + ".png"
        #names of files that depend on the map and the Tx:
        name2 = building_name + "_" + Tx_name + ".png"
        
        #Load buildings:

        img_name_buildings = os.path.join(self.dir_buildings, name1)
        image_buildings = Image.open(img_name_buildings).convert('L')
        
        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = Image.open(img_name_Tx).convert('L')
        
        # Load radio map:
        img_name_gain = os.path.join(self.dir_gain, name2)
        image_gain = Image.open(img_name_gain).convert('L')
        
        if self.free_prog:
            image_free_prog = os.path.join(self.dir_free_prog, name2)
            image_free_prog = Image.open(image_free_prog).convert('L')
        

        # Building complete for post processing
        self.post_buildings = os.path.join(self.input_dir, "buildings_complete/")
        build_comp_name = os.path.join(self.post_buildings, name1)
        build_comp = Image.open(build_comp_name).convert('L')

        seed = int(time.time() * 1000) % (2 ** 32)  # 使用当前时间戳生成种子
        if self.mixup_alpha > 0.0:
            mixup_alpha = self.mixup_alpha
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            Tx_name2 = str(random.randint(0, self.numTx-1))
            name2 = building_name + "_" + Tx_name2 + ".png"
            img_name_Tx2 = os.path.join(self.dir_Tx, name2)
            image_Tx2 = Image.open(img_name_Tx2).convert('L')
            img_name_gain2 = os.path.join(self.dir_gain, name2)
            image_gain2 = Image.open(img_name_gain2).convert('L')
            
        if self.free_prog:
            image_free_prog2 = os.path.join(self.dir_free_prog, name2)
            image_free_prog2 = Image.open(image_free_prog2).convert('L')
            
            
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            image_gain = self.transform(image_gain).type(torch.float32)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            image_buildings = self.transform(image_buildings).type(torch.float32)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            image_Tx = self.transform(image_Tx).type(torch.float32)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            build_comp = self.transform(build_comp).type(torch.float32)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if self.free_prog:
                image_free_prog = self.transform(image_free_prog).type(torch.float32)
            
            if self.mixup_alpha > 0.0:
                # TOM
                mixup_alpha = self.mixup_alpha
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                image_gain2 = self.transform(image_gain2).type(torch.float32)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                image_Tx2 = self.transform(image_Tx2).type(torch.float32)
                
                
                    
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                image_gain = lam * image_gain + (1 - lam) * image_gain2
                image_Tx = lam * image_Tx + (1 - lam) * image_Tx2
                
                if self.free_prog:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    image_free_prog2 = self.transform(image_free_prog2).type(torch.float32)
                    image_free_prog = lam * image_free_prog + (1 - lam) * image_free_prog2
                
        else:
            raise NotImplementedError

        if self.free_prog:
            inputs = torch.cat((image_buildings, image_Tx, image_free_prog), dim=0)
        else:
            inputs = torch.cat((image_buildings, image_Tx), dim=0)
        
        return [inputs, image_gain, build_comp, building_name, Tx_name]
    
    
if __name__ =='__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    x = RadioMapTestset(dataset_dir=r'/home/zzj/Pycharm_project/CUTS_Plus2/large-scale-channel-prediction/dataset', transform=trans)
    tmp = x[0]
    print(tmp[0].shape, tmp[1].shape, tmp[2].shape, tmp[3], tmp[4])

    gen = torch.Generator('cuda:1')
    laoder = DataLoader(x, batch_size=1, shuffle=True, num_workers=2, generator=gen)
    for i, (inputs, targets, build_comp, _, _) in enumerate(laoder):
        print(inputs.shape, targets.shape, build_comp.shape)
        break