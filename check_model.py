"""Quick script to check trained model"""
import torch

checkpoint = torch.load('checkpoints/best.pth', map_location='cpu', weights_only=False)
print("="*50)
print("TRAINED MODEL CHECKPOINT")
print("="*50)
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best Validation Loss: {checkpoint['best_val_loss']:.4f}")
print(f"Best Validation F1: {checkpoint.get('best_val_f1', 0):.4f}")
print(f"Checkpoint size: {len(checkpoint['model_state_dict'])} model components")
print("="*50)

