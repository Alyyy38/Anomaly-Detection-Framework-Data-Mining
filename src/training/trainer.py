"""
Training pipeline for anomaly detection framework
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json
import os

from ..models.framework import AnomalyDetectionFramework
from .losses import CombinedLoss
from ..config import Config


class Trainer:
    """Trainer class for anomaly detection framework"""
    
    def __init__(
        self,
        model: AnomalyDetectionFramework,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Anomaly detection model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        self.criterion = CombinedLoss(
            alpha=config.training.alpha,
            beta=config.training.beta,
            gamma=config.training.gamma,
            delta=config.training.delta
        )
        
        # Optimizers
        # Transformer optimizer
        transformer_params = list(self.model.transformer.parameters()) + \
                           list(self.model.contrastive.parameters())
        self.optimizer_transformer = optim.Adam(
            transformer_params,
            lr=config.training.transformer_lr,
            weight_decay=config.training.weight_decay
        )
        
        # GAN optimizers
        self.optimizer_generator = optim.RMSprop(
            self.model.generator.parameters(),
            lr=config.training.generator_lr
        )
        
        self.optimizer_discriminator = optim.RMSprop(
            self.model.discriminator.parameters(),
            lr=config.training.discriminator_lr
        )
        
        # Learning rate schedulers
        self.scheduler_transformer = self._create_scheduler(
            self.optimizer_transformer,
            config.training
        )
        self.scheduler_generator = self._create_scheduler(
            self.optimizer_generator,
            config.training
        )
        self.scheduler_discriminator = self._create_scheduler(
            self.optimizer_discriminator,
            config.training
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0
        
        # Create directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
    
    def _create_scheduler(self, optimizer, training_config):
        """Create learning rate scheduler"""
        if training_config.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_config.num_epochs
            )
        elif training_config.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=training_config.scheduler_step_size,
                gamma=training_config.scheduler_gamma
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'contrastive': 0.0,
            'gan_generator': 0.0,
            'gan_discriminator': 0.0,
            'gradient_penalty': 0.0
        }
        
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device)
            
            # Forward pass
            results = self.model(x, training=True, return_all=True)
            
            # Compute losses
            losses = self.criterion(
                results,
                x,
                discriminator=self.model.discriminator,
                training=True
            )
            
            # Update transformer and contrastive
            self.optimizer_transformer.zero_grad()
            transformer_loss = (
                self.config.training.alpha * losses['reconstruction'] +
                self.config.training.beta * losses['contrastive']
            )
            transformer_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.model.transformer.parameters(),
                self.config.training.gradient_clip_norm
            )
            self.optimizer_transformer.step()
            
            # Update generator
            self.optimizer_generator.zero_grad()
            gen_loss = self.config.training.gamma * losses['gan_generator']
            gen_loss.backward(retain_graph=True)
            self.optimizer_generator.step()
            
            # Update discriminator
            self.optimizer_discriminator.zero_grad()
            disc_loss = self.config.training.delta * losses['gan_discriminator']
            disc_loss.backward()
            self.optimizer_discriminator.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'recon': losses['reconstruction'].item(),
                'contr': losses['contrastive'].item()
            })
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses
    
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'contrastive': 0.0
        }
        
        all_scores = []
        all_labels = []
        
        for x, y in tqdm(self.val_loader, desc='Validation'):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            results = self.model(x, training=False)
            
            # Compute losses (no GAN losses during validation)
            losses = self.criterion(
                results,
                x,
                training=False
            )
            
            # Accumulate losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            
            # Collect scores and labels
            scores = results['anomaly_score'].cpu().numpy()
            labels = y.cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels)
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in total_losses:
            total_losses[key] /= num_batches
        
        # Compute metrics
        from ..utils.metrics import AnomalyMetrics
        metrics = AnomalyMetrics()
        threshold = np.median(all_scores)  # Simple threshold
        predictions = (np.array(all_scores) > threshold).astype(int)
        
        metrics_dict = metrics.compute_metrics(
            np.array(all_labels),
            predictions,
            np.array(all_scores)
        )
        
        total_losses.update(metrics_dict)
        
        return total_losses
    
    def train(self, num_epochs: Optional[int] = None):
        """Main training loop"""
        num_epochs = num_epochs or self.config.training.num_epochs
        
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses = self.validate_epoch()
            self.val_losses.append(val_losses)
            
            # Update schedulers
            if self.scheduler_transformer:
                self.scheduler_transformer.step()
            if self.scheduler_generator:
                self.scheduler_generator.step()
            if self.scheduler_discriminator:
                self.scheduler_discriminator.step()
            
            # Log to TensorBoard
            self._log_metrics(epoch, train_losses, val_losses)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f}")
            if 'f1_score' in val_losses:
                print(f"Val F1: {val_losses['f1_score']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                if 'f1_score' in val_losses:
                    self.best_val_f1 = val_losses['f1_score']
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Save final checkpoint
        self.save_checkpoint(self.current_epoch, is_best=False, final=True)
        self.writer.close()
    
    def _log_metrics(self, epoch: int, train_losses: Dict, val_losses: Dict):
        """Log metrics to TensorBoard"""
        # Training losses
        for key, value in train_losses.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        # Validation losses
        for key, value in val_losses.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Learning rates
        self.writer.add_scalar(
            'LR/transformer',
            self.optimizer_transformer.param_groups[0]['lr'],
            epoch
        )
        self.writer.add_scalar(
            'LR/generator',
            self.optimizer_generator.param_groups[0]['lr'],
            epoch
        )
        self.writer.add_scalar(
            'LR/discriminator',
            self.optimizer_discriminator.param_groups[0]['lr'],
            epoch
        )
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        final: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_transformer_state_dict': self.optimizer_transformer.state_dict(),
            'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
            'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            print(f"Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        # Save final
        if final:
            torch.save(checkpoint, self.checkpoint_dir / 'final.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_transformer.load_state_dict(checkpoint['optimizer_transformer_state_dict'])
        self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator_state_dict'])
        self.optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best val loss: {self.best_val_loss:.4f}")

