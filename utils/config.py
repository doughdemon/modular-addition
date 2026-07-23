import json

def load_cfg(task):
    cfg_file = open(f"tasks/{task}/config.json")
    cfg = json.load(cfg_file)

    seed = cfg['seed']
    frac_train = cfg['frac_train']
    layers = cfg['layers']
    lr = cfg['lr']
    group_size = cfg['group_size']
    weight_decay = cfg['weight_decay']
    betas = cfg['betas']
    num_epochs = cfg['num_epochs']
    return seed, frac_train, layers, lr, group_size, weight_decay, betas, num_epochs
