from abc import ABCMeta, abstractmethod
import os
from glob import glob
import time
import shutil
import logging
from datetime import datetime
import json
import inspect
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data
from tqdm import tqdm

from .seed import seed_everything
from .git import git, get_git_commit
from .log import BufferedReporter, to_rgb
from ..dataset.h5pan import H5PanDataset

# Import for evaluation
import evaluate_pansharpening

class SimplePanTrainer(metaclass=ABCMeta):
    cfg: dict
    
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    
    train_dataset: H5PanDataset
    val_dataset: H5PanDataset
    test_dataset: H5PanDataset
    
    train_loader: DataLoader
    val_loader: DataLoader
    
    out_dir: str
    
    disable_alloc_cache: bool
    
    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError
    
    @abstractmethod
    def _create_model(self, cfg):
        raise NotImplementedError
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(f"canconv.{cfg['exp_name']}")
        self.logger.setLevel(logging.INFO)
        seed_everything(cfg["seed"])
        self.logger.info(f"Seed set to {cfg['seed']}")
        
        self.dev = torch.device(cfg['device'])
        if self.dev.type != "cuda":
            raise ValueError(f"Only cuda device is supported, got {self.dev.type}")
        if self.dev.index != 0:
            self.logger.warning("Warning: Multi-GPU is not supported, the code may not work properly with GPU other than cuda:0. Please use CUDA_VISIBLE_DEVICES to select the device.")
            torch.cuda.set_device(self.dev)
            
        self.logger.info(f"Using device: {self.dev}")
        self._create_model(cfg)
        self.forward({
            'gt': torch.randn(cfg['batch_size'], cfg['spectral_num'], 64, 64),
            'ms': torch.randn(cfg['batch_size'], cfg['spectral_num'], 16, 16),
            'lms': torch.randn(cfg['batch_size'], cfg['spectral_num'], 64, 64),
            'pan': torch.randn(cfg['batch_size'], 1, 64, 64)
        })
        self.disable_alloc_cache = cfg.get("disable_alloc_cache", False)
        self.logger.info(f"Model loaded.")
        
        # Early stopping initializations
        self.early_stopping_patience = cfg.get("early_stopping_patience", 10) # Default patience
        # Prioritize validation loss for early stopping as requested
        self.early_stopping_metric_name = "val_loss" 
        self.early_stopping_delta = cfg.get("early_stopping_delta", 0.001) # How much change is considered an improvement
        
        self.best_primary_metric_val = float('inf') # For val_loss, lower is better
        
        # Track best values for specific ERGAS metrics we always want to consider for progress (can be logged separately)
        self.best_reduced_res_h5_ergas = float('inf')
        self.best_full_res_h5_ergas = float('inf')
        # Add MAT if needed
        self.best_reduced_res_mat_ergas = float('inf')
        self.best_full_res_mat_ergas = float('inf')

        self.early_stopping_counter = 0
        # Best model path remains tied to the primary metric improvement
        self.best_model_weights_path = os.path.join(self.out_dir if hasattr(self, 'out_dir') else os.path.join('runs', self.cfg["exp_name"]), "weights/best_model.pth")
        
    def _load_dataset(self):
        self.train_dataset = H5PanDataset(self.cfg["train_data"])
        self.val_dataset = H5PanDataset(self.cfg["val_data"])
        self.test_dataset = H5PanDataset(self.cfg["test_reduced_data"])
        
    def _create_output_dir(self):
        self.out_dir = os.path.join('runs', self.cfg["exp_name"])
        os.makedirs(os.path.join(self.out_dir, 'weights'), exist_ok=True)
        logging.info(f"Output dir: {self.out_dir}")
            
    def _dump_config(self):
        with open(os.path.join(self.out_dir, "cfg.json"), "w") as file:
            self.cfg["git_commit"] = get_git_commit()
            self.cfg["run_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
            json.dump(self.cfg, file, indent=4)
            
        try:
            source_path = inspect.getsourcefile(self.__class__)
            assert source_path is not None
            source_path = os.path.dirname(source_path)
            shutil.copytree(source_path, os.path.join(self.out_dir, "source"), ignore=shutil.ignore_patterns('*.pyc', '__pycache__'), dirs_exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to copy source code: ")
            self.logger.exception(e)
            
    def _on_train_start(self):
        pass
    
    def _on_val_start(self):
        pass
    
    def _on_epoch_start(self, epoch):
        pass
    
    @torch.no_grad()
    def run_test(self, dataset: H5PanDataset):
        self.model.eval()
        sr = torch.zeros(
            dataset.lms.shape[0], dataset.lms.shape[1], dataset.pan.shape[2], dataset.pan.shape[3], device=self.dev)
        for i in range(len(dataset)):
            sr[i:i+1] = self.forward(dataset[i:i+1])
        return sr

    @torch.no_grad()
    def run_test_for_selected_image(self, dataset, image_ids):
        self.model.eval()
        sr = torch.zeros(
            len(image_ids), dataset.lms.shape[1], dataset.pan.shape[2], dataset.pan.shape[3], device=self.dev)
        for i, image_id in enumerate(image_ids):
            sr[i:i+1] = self.forward(dataset[image_id:image_id+1])
        return sr
        
    def train(self):
        self._load_dataset()
        train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.cfg['batch_size'], shuffle=True, drop_last=False, pin_memory=True)
        val_loader = DataLoader(
            dataset=self.val_dataset, batch_size=self.cfg['batch_size'], shuffle=True, drop_last=False, pin_memory=True)
        self.logger.info(f"Dataset loaded.")
        
        self._create_output_dir()
        self._dump_config()
        self._on_train_start()
        
        writer = SummaryWriter(log_dir=self.out_dir)
        train_loss = BufferedReporter(f'train/{self.criterion.__class__.__name__}', writer)
        val_loss = BufferedReporter(f'val/{self.criterion.__class__.__name__}', writer)
        train_time = BufferedReporter('train/time', writer)
        val_time = BufferedReporter('val/time', writer)
        
        self.logger.info(f"Begin Training.")
        
        # Ensure paths for evaluation are available from cfg
        # These paths are typically like 'dataset/full_examples' or 'dataset/reduced_examples'
        # The evaluate_dataset function will look for .h5 or .mat files inside these dirs.
        self.full_res_h5_dir_eval = self.cfg.get('full_res_h5_dir_eval', self.cfg.get('test_origscale_data_dir')) # Try specific or fallback
        self.reduced_res_h5_dir_eval = self.cfg.get('reduced_res_h5_dir_eval', self.cfg.get('test_reduced_data_dir'))
        self.full_res_mat_dir_eval = self.cfg.get('full_res_mat_dir_eval')
        self.reduced_res_mat_dir_eval = self.cfg.get('reduced_res_mat_dir_eval')

        # Sensor specific params from cfg for evaluation
        self.sensor_range_max_eval = self.cfg.get('sensor_range_max', 2047.0)
        self.scale_ratio_eval = self.cfg.get('scale_ratio', 4)
        self.spectral_num_eval = self.cfg.get('spectral_num', 8) # Used if CANNet needs to be instantiated in eval script

        for epoch in tqdm(range(1, self.cfg['epochs'] + 1, 1)):
            self._on_epoch_start(epoch)
            
            self.model.train()
            for batch in tqdm(train_loader):
                start_time = time.time()
                
                self.model.zero_grad()
                sr, model_outputs = self.forward(batch)
                loss_val, loss_dict = self.criterion(sr, batch['gt'].to(self.dev), model_outputs)
                if isinstance(loss_val, tuple):
                    loss = loss_val[0]
                else:
                    loss = loss_val
                train_loss.add_scalar(loss.item())
                for loss_name, value in loss_dict.items():
                    train_loss.add_scalar(value.item(), name=loss_name)
                loss.backward()
                self.optimizer.step()
                
                if self.disable_alloc_cache:
                    torch.cuda.empty_cache()
                
                train_time.add_scalar(time.time() - start_time)
            train_loss.flush(epoch)
            train_time.flush(epoch)
            self.scheduler.step()
            self.logger.debug(f"Epoch {epoch} train done")
            
            if epoch % self.cfg['val_interval'] == 0:
                self._on_val_start()
                with torch.no_grad():
                    self.model.eval()
                    for batch in val_loader:
                        start_time = time.time()
                        sr, model_outputs = self.forward(batch)
                        loss_val, loss_dict = self.criterion(sr, batch['gt'].to(self.dev), model_outputs)
                        if isinstance(loss_val, tuple):
                            loss = loss_val[0]
                        else:
                            loss = loss_val
                        val_loss.add_scalar(loss.item())
                        for loss_name, value in loss_dict.items():
                            val_loss.add_scalar(value.item(), name=loss_name)
                        val_time.add_scalar(time.time() - start_time)
                    current_val_loss = val_loss.flush(epoch) # Capture returned average, this also writes to TB
                    val_time.flush(epoch)
                self.logger.debug(f"Epoch {epoch} val done")
                
                # --- Early Stopping Logic based on Validation Loss ---
                if current_val_loss is not None: # Ensure flush returned a valid number
                    if current_val_loss < self.best_primary_metric_val - self.early_stopping_delta:
                        self.logger.info(f"Validation loss improved from {self.best_primary_metric_val:.6f} to {current_val_loss:.6f}. Saving model.")
                        self.best_primary_metric_val = current_val_loss
                        self.early_stopping_counter = 0
                        os.makedirs(os.path.dirname(self.best_model_weights_path), exist_ok=True)
                        torch.save(self.model.state_dict(), self.best_model_weights_path)
                    else:
                        self.early_stopping_counter += 1
                        self.logger.info(f"Validation loss {current_val_loss:.6f} did not improve from {self.best_primary_metric_val:.6f} by at least {self.early_stopping_delta}. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                else:
                    self.logger.warning(f"Epoch {epoch}: Validation loss calculation returned None. Skipping early stopping check for this epoch.")

                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement in validation loss.")
                    break # Stop training
                
                # Force clean up memory after validation  
                if self.disable_alloc_cache:
                    torch.cuda.empty_cache()
            
            if epoch % self.cfg['checkpoint'] == 0 or ("save_first_epoch" in self.cfg and epoch <= self.cfg["save_first_epoch"]):
                torch.save(self.model.state_dict(), os.path.join(
                    self.out_dir, f'weights/{epoch}.pth'))
                self.logger.info(f"Epoch {epoch} checkpoint saved")
        
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "weights/final.pth"))
        self.logger.info(f"Training finished.")
        writer.close() # Close writer at the end of training