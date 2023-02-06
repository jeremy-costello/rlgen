# imports
import os
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, train_config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_config = train_config
    
    def setup(self, rank):
        os.environ['MASTER_ADDR'] = self.train_config.master_address
        os.environ['MASTER_PORT'] = self.train_config.master_port
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = self.train_config.torch_distributed_debug

        dist.init_process_group(self.train_config.backend,
                                rank=rank, world_size=self.train_config.world_size)
    
    @staticmethod
    def cleanup():
        dist.destroy_process_group()
    
    def distributed_train(self):
        mp.spawn(self.train, nprocs=self.train_config.world_size, join=True)
    
    def save_checkpoint(self, model, epoch):
        ckpt_file = f'{self.train_config.ckpt_path}/model_{epoch}.ckpt'

        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), ckpt_file)
    
    def train(self, rank):
        model, train_config = self.model, self.train_config
        scaler = GradScaler(enabled=train_config.mixed_precision)
        self.setup(rank)
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=train_config.find_unused_parameters)
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(train_config)
        if train_config.lr_scheduler is not None:
            lr_scheduler = raw_model.configure_lr_schedulers(train_config)
        
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            distributed_sampler = DistributedSampler(data, shuffle=is_train)

            if train_config.deterministic_dataloader:
                def seed_worker(worker_id):
                    worker_seed = torch.initial_seed() % 2**32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)
                
                generator = torch.Generator()
                generator.manual_seed(train_config.seed)

                worker_init_fn = seed_worker
            else:
                worker_init_fn = None
                generator = None
            
            loader = DataLoader(data, pin_memory=True,
                                batch_size=train_config.batch_size,
                                num_workers=train_config.num_workers,
                                worker_init_fn=worker_init_fn,
                                generator=generator,
                                sampler=distributed_sampler)
            
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for _, batch in pbar:
                batch_start = time.perf_counter()

                with torch.set_grad_enabled(is_train):
                    with torch.autocast('cuda', enabled=train_config.mixed_precision):
                        loss = model(batch)
                
                # could put m-step loop from MPO here? or have this and a bit higher in separate file
                if is_train:
                    scaler.scale(loss).backward()
                    # all below should only run on on gradient accumulation step
                    if train_config.grad_norm_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    if train_config.lr_scheduler is not None:
                        lr_scheduler.step()
                    model.zero_grad()
                
                batch_end = time.perf_counter()
                steps_per_second = 1 / (batch_end - batch_start)
                rank_samples_per_second = train_config.batch_size * steps_per_second
                global_samples_per_second = rank_samples_per_second * train_config.world_size
                samples_seen = global_step * train_config.batch_size * train_config.world_size
                
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / train_config.world_size

                if is_train and rank == 0:
                    pbar.update(1)
                    global_step += 1
                    learning_rate = train_config.learning_rate if train_config.lr_scheulder is None \
                        else lr_scheduler.get_last_lr()[0]
                    logs = {
                        'train/loss': loss.detach().item(),
                        'train/lr': learning_rate,
                        'train/epoch': epoch,
                        'train/step': global_step,
                        'train/samples_seen': samples_seen,
                        'perf/rank_samples_per_second': rank_samples_per_second,
                        'perf/global_samples_per_second': global_samples_per_second
                    }
                    pbar.set_postfix(logs)
                        
            return loss
        
        global_step = 0
        best_loss = float('inf')
        for epoch in range(train_config.epochs):
            _ = run_epoch('train')
            if self.test_dataset is None:
                if rank == 0:
                    self.save_checkpoint(model)
            else:
                test_loss = run_epoch('test')
                if test_loss < best_loss:
                    best_loss = test_loss
                    if rank == 0:
                        self.save_checkpoint(model)
        
        self.cleanup()
