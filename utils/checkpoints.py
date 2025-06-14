import torch

def save_checkpoint(model, optimizer, loss, task_dir, epoch=None, final=False):
    """
    Save model checkpoint.
    """
    path = f'{task_dir}/checkpoints/epoch_{epoch}.pt'
    if final:
        path = f'{task_dir}/model.pt'
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
	'loss': loss
	}, path)

def load_checkpoint(model, optimizer, task_dir, epoch=None, final=False):
    """
    Load model checkpoint.
    """
    path = f'{task_dir}/checkpoints/epoch_{epoch}.pt'
    if final:
        path = f'{task_dir}/model.pt'
    checkpoint = torch.load(path)

    if model: model.load_state_dict(checkpoint['model'])
    if optimizer: optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch'], checkpoint['loss']
