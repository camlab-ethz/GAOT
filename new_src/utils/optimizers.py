import torch
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass

###############
# Config
###############
@dataclass
class OptimizerArgsConfig:
    lr: float = 1e-3                    # Initial learning rate
    weight_decay: float = 1e-3          # L2 regularization (weight decay) coefficient
    epoch: int = 100                    # Total number of training epochs
    loss_scale: float = 1.0             # Loss scaling factor 
    eval_every_eps: int = 2             # Evaluate every n epochs
    scheduler: str = "mix"              # Learning rate scheduler type, support ['step', 'cos', 'exp', 'mix']
    early_save_metric: str = 'val'      # Metric for early stopping, support ['train', 'val']
    # for mix scheduler
    max_lr: float = 1e-2                # Maximum learning rate for the cosine annealing phase
    min_lr: float = 1e-5                # Minimum learning rate for the cosine annealing phase
    final_lr: float = 1e-5              # Final learning rate for the exponential decay phase
    # for step scheduler
    scheduler_step_size: int = 100      # Step size (number of epochs) for StepLR scheduler
    scheduler_gamma: float = 0.8        # Multiplicative factor for learning rate decay in StepLR scheduler
    scheduler_T_max: int = 100          # Maximum number of iterations (usually total epochs) for CosineAnnealingLR scheduler
    scheduler_eta_min: float = 1e-4     # Minimum learning rate for CosineAnnealingLR scheduler

###############
# Scheduler
###############
class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, cosine_epochs, exp_decay_epochs,
                 initial_lr, max_lr, min_lr, final_lr, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = cosine_epochs
        self.exp_decay_epochs = exp_decay_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.final_lr = final_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # first phase (warm up): initial_lr to max_lr
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.last_epoch / max(1, self.warmup_epochs - 1))
        elif self.last_epoch < self.warmup_epochs + self.cosine_epochs:
            # second stage (cosine): max_lr to min_lr
            epoch = self.last_epoch - self.warmup_epochs
            cosine_ratio = (1 + np.cos(np.pi * epoch / self.cosine_epochs)) / 2
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_ratio
        else:
            # third stage (expontential)： min_lr to final_lr
            epoch = self.last_epoch - self.warmup_epochs - self.cosine_epochs
            decay_steps = max(1, self.exp_decay_epochs - 1)
            lr = self.min_lr * ((self.final_lr / self.min_lr) ** (epoch / decay_steps))
        return [lr for _ in self.optimizer.param_groups]

###############
# Optimizer
###############
class AdamOptimizer:
    optimizer: torch.optim.Adam
    scheduler: torch.optim.lr_scheduler.StepLR
    epoch: int
    batch_size: int
    lr: float   
    loss_scale: float
    eval_every_eps: int

    def __init__(self, params, config):
        self.optimizer = torch.optim.Adam(params, lr=config.lr)
        self.epoch = config.epoch
        self.lr = config.lr  
        self.loss_scale = config.loss_scale
        self.eval_every_eps = config.eval_every_eps
        self.early_save_metric = config.early_save_metric.lower()

        if self.early_save_metric not in ['train', 'val']:
            raise ValueError("`early_save_metric` must be either 'train' or 'val'.")

        if config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        elif config.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.scheduler_T_max, eta_min=config.scheduler_eta_min)
        elif config.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.scheduler_gamma)
        elif config.scheduler == 'mix':
            warmup_epochs = int(0.02 * self.epoch)
            cosine_epochs = int(0.96 * self.epoch)
            exp_decay_epochs = self.epoch - warmup_epochs - cosine_epochs
            if warmup_epochs == 0:
                warmup_epochs = 1
                cosine_epochs -= 1
            if exp_decay_epochs == 0:
                exp_decay_epochs = 1
                cosine_epochs -= 1
            self.scheduler = CustomLRScheduler(
                optimizer=self.optimizer,
                total_epochs=self.epoch,
                warmup_epochs=warmup_epochs,
                cosine_epochs=cosine_epochs,
                exp_decay_epochs=exp_decay_epochs,
                initial_lr=self.lr,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                final_lr=config.final_lr
            )
        else:
            self.scheduler = None

    def optimize(self, trainer: 'BaseTrainer',
                        description: str = "AdamOptimizer",
                        color: str = "blue"):
        time_total = 0.0
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        epochs = []
        val_epochs = []
        val_losses = []

        pbar = tqdm(total=self.epoch, desc=description, colour = color)
        for epoch in range(self.epoch):
            trainer.model.train()
            total_loss = 0.0
            for batch in trainer.train_loader:
                start_time = time.time()
                self.optimizer.zero_grad()
                train_loss = trainer.train_step(batch)
                train_loss.backward()
                self.optimizer.step()
                total_loss += train_loss.item()
                if trainer.device.type == 'cuda':
                    torch.cuda.synchronize()
                time_total += time.time() - start_time
            
            if self.scheduler is not None:
                self.scheduler.step()
        
            pbar.update(1)

            if (epoch + 1) % self.eval_every_eps == 0:
                train_loss = total_loss / len(trainer.train_loader)
                losses.append(train_loss)
                epochs.append(epoch)
                val_loss = trainer.validate(trainer.val_loader)
                pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})
                val_losses.append(val_loss)
                val_epochs.append(epoch)

                if self.early_save_metric == 'val':
                    current_loss = val_loss
                else:
                    current_loss = train_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    best_state = deepcopy(trainer.model.state_dict())
        
        if best_state is not None:
            trainer.model.load_state_dict(best_state)

        pbar.close()
        
        return {
            "train":{
                "loss": losses,
                "epoch": epochs,
            },
            "valid":{
                "loss": val_losses,
                "epoch": val_epochs,
            },
            "best":{
                "epoch": best_epoch,
                "loss": best_loss,
            },
            "time": time_total
        }

class AdamWOptimizer:
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler._LRScheduler
    epoch: int
    batch_size: int
    lr: float
    loss_scale: float
    eval_every_eps: int

    def __init__(self, params, config):
        self.optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        self.epoch = config.epoch
        self.lr = config.lr  
        self.loss_scale = config.loss_scale
        self.eval_every_eps = config.eval_every_eps
        self.early_save_metric = config.early_save_metric.lower()

        if self.early_save_metric not in ['train', 'val']:
            raise ValueError("`early_save_metric` must be either 'train' or 'val'.")
        
        if config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        elif config.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.scheduler_T_max, eta_min=config.scheduler_eta_min)
        elif config.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.scheduler_gamma)
        elif config.scheduler == 'mix':
            warmup_epochs = int(0.02 * self.epoch)
            cosine_epochs = int(0.90 * self.epoch)
            exp_decay_epochs = self.epoch - warmup_epochs - cosine_epochs
            if warmup_epochs == 0:
                warmup_epochs = 1
                cosine_epochs -= 1
            if exp_decay_epochs == 0:
                exp_decay_epochs = 1
                cosine_epochs -= 1
            self.scheduler = CustomLRScheduler(
                optimizer=self.optimizer,
                total_epochs=self.epoch,
                warmup_epochs=warmup_epochs,
                cosine_epochs=cosine_epochs,
                exp_decay_epochs=exp_decay_epochs,
                initial_lr=self.lr,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                final_lr=config.final_lr
            )
        else:
            self.scheduler = None

    def optimize(self, trainer: 'BaseTrainer',
                 description: str = "AdamWOptimizer",
                 color: str = "blue"):
        time_total = 0.0
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        epochs = []
        val_epochs = []
        val_losses = []

        pbar = tqdm(total=self.epoch, desc=description, colour=color)
        for epoch in range(self.epoch):
            trainer.model.train()
            total_loss = 0.0
            for batch in trainer.train_loader:
                start_time = time.time()
                self.optimizer.zero_grad()
                train_loss = trainer.train_step(batch)
                train_loss.backward()
                self.optimizer.step()

                total_loss += train_loss.detach()
                # if trainer.device.type == 'cuda':
                #     torch.cuda.synchronize()
                # time_total += time.time() - start_time

            if self.scheduler is not None:
                self.scheduler.step()

            pbar.update(1)

            if (epoch + 1) % self.eval_every_eps == 0:
                train_loss = total_loss.cpu().item() / len(trainer.train_loader)
                losses.append(train_loss)
                epochs.append(epoch)
                val_loss = trainer.validate(trainer.val_loader)
                pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})
                val_losses.append(val_loss)
                val_epochs.append(epoch)

                if self.early_save_metric == 'val':
                    current_loss = val_loss
                else:
                    current_loss = train_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    best_state = deepcopy(trainer.model.state_dict())

        if best_state is not None:
            trainer.model.load_state_dict(best_state)

        pbar.close()

        return {
            "train": {
                "loss": losses,
                "epoch": epochs,
            },
            "valid": {
                "loss": val_losses,
                "epoch": val_epochs,
            },
            "best": {
                "epoch": best_epoch,
                "loss": best_loss,
            },
            "time": time_total
        }  