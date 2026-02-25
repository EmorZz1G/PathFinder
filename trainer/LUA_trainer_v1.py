import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from models import load_model_from_yaml
import torch.optim as optim

from utils.misc import reproduc
from datas.dataloader import RadioMapTestset,eval_model,inference_all_images_cal_loss,Metrics
from time import time, strftime, localtime
import numpy as np
import os
import torchvision.transforms as transforms


# DynamicHuberLoss or MomentumPredictionLoss
class MomentumPredictionLoss(torch.nn.Module):
    def __init__(self, initial_delta=1.0, update_factor=0.9):
        super(MomentumPredictionLoss, self).__init__()
        self.delta = torch.nn.Parameter(torch.tensor(initial_delta))
        self.update_factor = update_factor

    def forward(self, outputs, targets):
        error = torch.abs(outputs - targets)
        huber_loss = torch.where(error < self.delta, 0.5 * (outputs - targets) ** 2, self.delta * (error - 0.5 * self.delta))
        mean_error = torch.mean(error)
        self.delta.data = torch.max(self.delta * self.update_factor, mean_error)
        self.delta.data = torch.clamp(self.delta, 1e-6, 10)
        # huber_loss = torch.sqrt(huber_loss)
        x = torch.mean(huber_loss)
        return x
 
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))
    
def NMSELoss(pred, target):
    return torch.sum((pred - target)**2) / torch.sum(target**2)
    
def dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss
    
    
class EarlyStopping:
    def __init__(self, config, patience=90, verbose=False, delta=0, path='checkpoint3.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.config = config
        
        self.save_path = config.save_path
        self.model_pth = config.model_path
        
        base_pth = os.path.dirname(__file__)
        parent_pth = os.path.dirname(base_pth)
        self.model_pth = os.path.join(parent_pth, self.save_path, self.model_pth)
        print(f'model_pth: {self.model_pth}')
        os.makedirs(self.model_pth, exist_ok=True)
        
    def load_checkpoint(self, model):
        pth = os.path.join(self.model_pth, self.path)
        model.load_state_dict(torch.load(pth))
        print(f'Checkpoint loaded from {pth}')
        
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        pth = os.path.join(self.model_pth, self.path)
        torch.save(model.module.state_dict(), pth)
        self.val_loss_min = val_loss
            
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        pth = os.path.join(self.model_pth, self.path)
        pth = pth.replace('.pt', f'_last.pt')
        torch.save(model.module.state_dict(), pth)
        
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self, config):
        # 使用示例
        self.model_param = config.model
        
        self.criterion = torch.nn.MSELoss()
        
        if config.bias_loss > 0:
            self.bias_criterion = torch.nn.MSELoss()
            self.bias_ceof = config.bias_loss
        else:
            self.bias_criterion = None
            self.bias_ceof = 0

        model_name = config.model.name
        datetime = strftime("%Y-%m-%d-%H-%M-%S", localtime())
        self.logger = SummaryWriter(log_dir=os.path.join(config.save_path, config.log_path, f'{model_name}_{datetime}'))
        self.logger.add_text('config', str(config))
        
        self.dice_loss = None
        
        self.metric = RMSELoss()
        self.config = config
        self.device = config.device_ids[0]
        reproduc(**config.reproduc)
        self.early_stopping = EarlyStopping(config, verbose=True, path=self.config.ckp_name)
        
        self.build_model()
        if self.config.opti.name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.opti.lr, weight_decay=self.config.opti.weight_decay)
        elif self.config.opti.name == 'AdamW':
            try:
                betas = self.config.opti.betas
            except:
                betas = (0.9, 0.99)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.opti.lr, weight_decay=self.config.opti.weight_decay, betas=betas)
            
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.build_dataloader()
        
    def build_dataloader(self):
        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None
        
        train_idx = self.config.dataset.train_idx
        
        try:
            rnd_train = self.config.dataset.rnd_train
        except:
            rnd_train = False
            
        try:
            mixup_alpha = self.config.dataset.mixup_alpha
        except:
            mixup_alpha = 0.0
            
        try:
            free_prog = self.config.dataset.free_prog
        except:
            free_prog = False
        
        
        print('Building DataLoader... {}'.format(self.device))
        generator = torch.Generator(self.device)

        # 定义数据增强操作，这里使用Compose将多个变换组合在一起
        train_transform = transforms.Compose([
            # 以0.5的概率进行随机水平翻转（左右对称翻转）
            transforms.RandomHorizontalFlip(p=0.5),  
            # 以0.5的概率进行随机垂直翻转（上下对称翻转）
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=(-4,4)),
            # 将图像数据转换为张量，这是后续在深度学习模型中使用的标准格式
            transforms.ToTensor(),  
        ])
        
        print('Use data augmentation')
        dataset0 = RadioMapTestset(train_idx[0], train_idx[1], dataset_dir=self.config.dataset.dataset_dir, transform=train_transform, rnd_train=rnd_train, mixup_alpha=mixup_alpha, free_prog=free_prog)
        # dataset0 = RadioMapTestset(train_idx[0], train_idx[1], dataset_dir=self.config.dataset.dataset_dir)
        samepler0 = DistributedSampler(dataset0, shuffle=True)
        self.train_loader = DataLoader(dataset0, batch_size=self.config.opti.batch_size, shuffle=False, num_workers=self.config.opti.num_workers, 
                                    #    generator=generator, 
                                       sampler=samepler0)
        test_idx = self.config.dataset.test_idx
        dataset1 = RadioMapTestset(test_idx[0], test_idx[1], dataset_dir=self.config.dataset.dataset_dir, rnd_train=rnd_train,  free_prog=free_prog)
        samepler1 = DistributedSampler(dataset1,shuffle=True)
        self.test_loader = DataLoader(dataset1, batch_size=self.config.opti.batch_size, shuffle=False, num_workers=self.config.opti.num_workers, 
                                    #   generator=generator, 
                                    sampler=samepler1
                                      )
        valid_idx = self.config.dataset.valid_idx
        dataset2 = RadioMapTestset(valid_idx[0], valid_idx[1], dataset_dir=self.config.dataset.dataset_dir, rnd_train=rnd_train , free_prog=free_prog)
        samepler2 = DistributedSampler(dataset2,shuffle=True)
        self.valid_loader = DataLoader(dataset2, batch_size=self.config.opti.batch_size, shuffle=False, num_workers=self.config.opti.num_workers,
                                    #    generator=generator, 
                                        sampler=samepler2
                                       )
        print('DataLoader has been built')

    def build_model(self):
        # 2） 配置每个进程的gpu
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        print(f'local_rank: {local_rank}', f'device: {self.device}')
        self.model = load_model_from_yaml(self.config)
        self.model.to(self.device)
        if self.config.load_pretrain:
            self.early_stopping.load_checkpoint(self.model)
        # self.model = torch.nn.DataParallel(self.model, device_ids=self.config.device_ids)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
        print('Model has been built')
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        loader_len = len(train_loader)
        st = time()
        
        train_loss_avg = (0, 0)
        self.train_loader.sampler.set_epoch(epoch)
        self.test_loader.sampler.set_epoch(epoch)
        self.valid_loader.sampler.set_epoch(epoch)
        
        
        from collections import defaultdict
        metrics_list = defaultdict(list)
        N_samples = 0
        for i, (data, target, build, *_) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output, bias = self.model(data)
            mse_loss = self.criterion(output, target)
            
            if self.bias_criterion is not None:
                bias_loss = self.bias_criterion(bias, data)
            else:
                bias_loss = torch.zeros(1).to(self.device)
            
            loss = mse_loss + bias_loss * self.bias_ceof
            
            train_loss_avg = (train_loss_avg[0] + mse_loss.item(), train_loss_avg[1] + 1)
            N_samples += data.shape[0]
            loss.backward()
            self.optimizer.step()
            RMSE = mse_loss.item()**0.5
            NMSE = NMSELoss(output, target).detach().item()
            if i % 10 == 0:
                print(f'Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tMSE: {mse_loss.item():.8f}\tRMSE: {RMSE:.8f}\t BiasLoss: {bias_loss.item():.8f} \tTime used/needed: [{time()-st:.2f} / {(time()-st)/(i+1)*(loader_len-i-1):.2f}]')
            
            if self.local_rank == 0:
                self.logger.add_scalar('Train-Iter/MSE', mse_loss.item(), epoch*loader_len+i)
                self.logger.add_scalar('Train-Iter/RMSE', RMSE, epoch*loader_len+i)
                self.logger.add_scalar('Train-Iter/NMSE', NMSE, epoch*loader_len+i)
                self.logger.add_scalar('Train-Iter/BiasLoss', bias_loss.item(), epoch*loader_len+i)
                
                metrics = Metrics(output, target, build)
                self.logger.add_scalar('Train-Iter/MSE-mask', metrics.MSE, epoch*loader_len+i)
                self.logger.add_scalar('Train-Iter/RMSE-mask', metrics.RMSE, epoch*loader_len+i)
                self.logger.add_scalar('Train-Iter/NMSE-mask', metrics.NMSE, epoch*loader_len+i)
                
                metrics_list['MSE-mask'].append(metrics.MSE * data.shape[0])
                metrics_list['RMSE-mask'].append(metrics.RMSE * data.shape[0])
                metrics_list['NMSE-mask'].append(metrics.NMSE * data.shape[0])
                metrics_list['MSE'].append(mse_loss.item() * data.shape[0])
                metrics_list['RMSE'].append(RMSE * data.shape[0])
                metrics_list['NMSE'].append(NMSE * data.shape[0])
                
                
            valid_step_interval = self.config.opti.valid_step_interval
            valid_steps = np.arange(0, loader_len, loader_len//valid_step_interval)
            if valid_step_interval is not None and i in valid_steps and self.local_rank == 0:
                self.valid_epoch(self.valid_loader, epoch)
                self.model.train()
                if self.early_stopping.early_stop:
                    print("Early stopping !!")
                    break
        
        if self.local_rank == 0:
            for k, v in metrics_list.items():
                metrics_list[k] = sum(v) / N_samples
                
            self.logger.add_scalar(f'Train/MSE-mask', metrics_list['MSE-mask'], epoch)
            self.logger.add_scalar(f'Train/RMSE-mask', metrics_list['RMSE-mask'], epoch)
            self.logger.add_scalar(f'Train/NMSE-mask', metrics_list['NMSE-mask'], epoch)
            self.logger.add_scalar(f'Train/MSE', metrics_list['MSE'], epoch)
            self.logger.add_scalar(f'Train/RMSE', metrics_list['RMSE'], epoch)
            self.logger.add_scalar(f'Train/NMSE', metrics_list['NMSE'], epoch)
                
        train_loss_avg = (train_loss_avg[0] / train_loss_avg[1])
        rmse = train_loss_avg**0.5
        print(f'Train Epoch: {epoch}\t Average MSE loss: {train_loss_avg:.8f} RMSE loss: {rmse:.8f} Time used: {time()-st:.2f}')
                
    
    def log_image(self, imgs, targets, epoch, val_loss):
        
        def plot2(x,y,pth):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            im1 = ax1.imshow(x, cmap='gray')
            im2 = ax2.imshow(y, cmap='gray')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im2, cax=cax)
            fig.savefig(pth)
            
        
        if hasattr(self, 'log_cnt'):
            self.log_cnt += 1
        else:
            self.log_cnt = 0
        log_path = os.path.join(self.config.save_path, self.config.log_path, 'log_img')
        os.makedirs(log_path, exist_ok=True)
        ckpt = self.config.ckp_name if self.config.ckp_name is not None else 'ckpt'
        
        cnt = self.log_cnt  
        file_name = f'epoch{epoch}_{ckpt}_cnt{cnt}_E_{epoch}_RMSE_{val_loss:.6f}.png'
        x = imgs[0][0].detach().cpu().numpy()
        y = targets[0][0].detach().cpu().numpy()
        # save predicted image
        from matplotlib import pyplot as plt
        plot2(x, y, os.path.join(log_path, file_name))
        print(f'Image saved: {file_name}')
                    
    
    @torch.no_grad()
    def valid_epoch(self, valid_loader, epoch, log_img=True, logging=False, post_process=True):
        N_samples = 0
        valid_loss = 0
        nmse_loss = 0
        self.model.eval()
        from collections import defaultdict
        metircs_list = defaultdict(list)
        for i, (data, target, build , *_) in enumerate(valid_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            if post_process:
                thre = 1/255
                output[output<thre] = 0
            valid_loss += self.metric(output, target).item() * data.shape[0]
            nmse_loss += NMSELoss(output, target).item() * data.shape[0]
            N_samples += data.shape[0]
            
            if logging:
                metrics = Metrics(output, target, build)
                metircs_list['MSE-mask'].append(metrics.MSE * data.shape[0])
                metircs_list['RMSE-mask'].append(metrics.RMSE * data.shape[0])
                metircs_list['NMSE-mask'].append(metrics.NMSE * data.shape[0])
            
        valid_loss /= N_samples
        nmse_loss /= N_samples
        print(f'Valid: {epoch}\t Valid set: Average RMSE loss: {valid_loss:.8f}\t MSE loss: {valid_loss**2:.8f}\t NMSE loss: {nmse_loss:.8f}')
        
        if logging:
            for k, v in metircs_list.items():
                metircs_list[k] = sum(v) / N_samples
                print(f'{k}: {metircs_list[k]:.8f}', end='\t')
            
            self.logger.add_scalar('Valid/MSE-mask', metircs_list['MSE-mask'], epoch)
            self.logger.add_scalar('Valid/RMSE-mask', metircs_list['RMSE-mask'], epoch)
            self.logger.add_scalar('Valid/NMSE-mask', metircs_list['NMSE-mask'], epoch)
            
            self.logger.add_scalar('Valid/RMSE', valid_loss, epoch)
            self.logger.add_scalar('Valid/MSE', valid_loss**2, epoch)
            self.logger.add_scalar('Valid/NMSE', nmse_loss, epoch)
        
        if log_img and self.local_rank == 0:
            self.log_image(output, target, epoch, valid_loss)
        
        if self.local_rank == 0:
            self.early_stopping(valid_loss, self.model)
        return valid_loss
        
    @torch.no_grad()
    def test_epoch(self, test_loader, epoch, post_process=True):
        self.model.eval()
        test_loss = 0
        
        N_samples = 0
        for i, (data, target, *_) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            if post_process:
                thre = 1/255
                output[output<thre] = 0
            test_loss += self.metric(output, target).item() * data.shape[0]
            N_samples += data.shape[0]
                
        test_loss /= N_samples
        print('\nTest set: Average loss: {:.8f}\n'.format(test_loss))
        return test_loss

        
    def train(self, train_loader=None, valid_loader=None, test_loader=None):
        if train_loader is None:
            train_loader = self.train_loader
        if valid_loader is None:
            valid_loader = self.valid_loader
        if test_loader is None:
            test_loader = self.test_loader
            
        
        for epoch in range(self.config.epochs):
            self.model.train()
            self.train_epoch(train_loader, epoch)
            if self.early_stopping.early_stop:
                print("Early stopping !!")
                break
            self.scheduler.step()
            
            if test_loader is not None and epoch % self.config.opti.test_epoch == 0 and self.local_rank == 0:
                self.test_epoch(test_loader, epoch)
            
        if self.test_loader is not None and self.local_rank == 0:
            self.test_epoch(test_loader, epoch)
            
            
    def inference(self, test_loader=None, output_img=False):
        test_loader = self.test_loader if test_loader is None else test_loader
        output_dir = os.path.join(self.config.save_path, 'inference')
        os.makedirs(output_dir, exist_ok=True)
        print(f'Inference output dir: {output_dir}')
        loss = inference_all_images_cal_loss(self.model, test_loader, output_dir, output_img)
        self.model.eval()
        loss = eval_model(self.model, test_loader)
        print(f'Inference loss: {loss}')