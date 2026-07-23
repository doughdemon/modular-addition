import os
from pathlib import Path
import torch

def save_checkpoint(model, optimizer, loss, task, epoch=None, final=False):
    """
    Save model checkpoint.
    """
    base_path = Path(f'outputs/{task}')
    checkpoint_path = base_path / 'checkpoints'
    checkpoint_path.makedir(parents=True, exist_ok=True)
    path = checkpoint_path / f'epoch_{epoch}.pt'
    if final:
        path = base_path / 'model.pt'
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
	'loss': loss
	}, path)

def load_checkpoint(model, optimizer, task, epoch=None, final=False):
    """
    Load model checkpoint.
    """
    base_path = Path(f'outputs/{task}')
    path = base_path / f'checkpoints/epoch_{epoch}.pt'
    if final:
        path = base_path / f'model.pt'
    checkpoint = torch.load(path)

    if model: model.load_state_dict(checkpoint['model'])
    if optimizer: optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch'], checkpoint['loss']
