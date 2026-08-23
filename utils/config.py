import json

def load_cfg(task):
    cfg_file = open(f"tasks/{task}/config.json")
    cfg = json.load(cfg_file)

    group_size = cfg['group_size']

    model_seed = cfg['model_seed']
    layers = cfg['layers']
    activation = cfg['activation']

    data_seed = cfg['data_seed']
    frac_train = cfg['frac_train']

    lr = cfg['lr']
    weight_decay = cfg['weight_decay']
    betas = cfg['betas']
    num_epochs = cfg['num_epochs']

    return model_seed, data_seed, frac_train, layers, activation, lr, group_size, weight_decay, betas, num_epochs
