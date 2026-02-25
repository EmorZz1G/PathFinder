import argparse
import os
import json

import torch
import torch.multiprocessing as mp

from torch.distributed import init_process_group

# if not torch.cuda.is_initialized():
#     torch.cuda.init()

def ddp_setup(rank, world_size, port=23499):
    """
    Args:
        rank: 进程的唯一标识，在 init_process_group 中用于指定当前进程标识
        world_size: 进程总数
    """
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method=f'tcp://127.0.0.1:{port}')
    torch.cuda.set_device(rank)

args = argparse.ArgumentParser()
import yaml
args.add_argument('--version', type=str, default='default')
args.add_argument('--config_pth', type=str, default='config')
args.add_argument('--config', type=str, default='default.yaml')
args.add_argument('--mode', type=str, default='train')
args.add_argument('--port', type=int, default=23498)

args = args.parse_args()

class Config(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getstate__(self):
        return dict(self)  # 返回字典自身的键值对，作为序列化的状态信息

    def __setstate__(self, state):
        self.clear()  # 先清空当前对象的内容
        self.update(state)  # 用传入的状态字典更新自身内容，恢复对象状态

    
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)



# from trainer.trainer_v1 import Trainer
# from trainer.trainer_v2 import Trainer
# from trainer.LUA_trainer_v1 import Trainer
from trainer.UG_trainer_v1 import Trainer
from torch.distributed import init_process_group, destroy_process_group
import yaml

def main(rank, world_size, config, port):
    
    ddp_setup(rank, world_size, port)
    trainer = Trainer(config)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.inference()
    destroy_process_group()

if __name__ == '__main__': 
    world_size = torch.cuda.device_count()
    config = load_config(os.path.join(args.config_pth, args.config))
    config = Config(config)
    print("Config loaded")
    print(config)
    mp.spawn(main, args=(world_size, config, args.port), nprocs=world_size)
    
    if args.mode == 'train':
        print("Training finished.")
    elif args.mode == 'test':
        print("Inference finished.")
    
    
