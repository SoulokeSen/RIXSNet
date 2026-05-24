#import sidechainnet as scn

import engineer

from engineer.schedulers.cosine import CosineAnnealingLR
import torch
import utils
torch.set_default_dtype(torch.float64)

def main(config):
    dataset_config = config["dataset"]
    batch_size = dataset_config["batch_size"]
    n_train = dataset_config["n_train"]
    dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)
    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()
#    exit()
#    traindebug_loader = dataset.traindebug_loader()
    
    model_config = config["model"]
    model = engineer.load_module(model_config.pop("module"))(**model_config)

#    model = model.cuda()
    optimizer_config = config["optimizer"]
    min_lrs = optimizer_config.pop("min_lrs")
    decay_steps = optimizer_config.pop("decay_steps")
    
    optimizer = engineer.load_module(optimizer_config.pop("module"))(
        model.parameters(), **optimizer_config
    )

    # scheduler_config = config['scheduler']
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode=scheduler_config["mode"], 
    #     factor=scheduler_config["factor"], 
    #     patience=scheduler_config["patience"])

    prntinterval = int(n_train/batch_size)
    print("printinteral", prntinterval)
    loss_config = config["loss"]
    loss_config["printstep"] = prntinterval
    loss = engineer.load_module(loss_config.pop("module"))(**loss_config)
    
    steps = config["trainer"]["max_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = CosineAnnealingLR(
        optimizer,
        steps,
        warmup_steps=int(1 / 32 * steps),
        decay_steps=int(decay_steps * steps),
        min_lrs=min_lrs
    )
    # scheduler=None

    trainer_module = engineer.load_module(config["trainer"].pop("module"))

    trainer_config = config["trainer"]
    trainer_config['run_dir'] = config['run_dir']
    trainer_config['use_wandb'] = 'wandb' in config
    trainer_config['val_check_interval'] = prntinterval
    trainer_config['log_interval'] = prntinterval


    trainer = trainer_module(
        **trainer_config,
    )
    train_loss, val_loss = trainer.fit(model, optimizer, loss, train_loader, scheduler, val_loader, test_loader=test_loader)

    loss_matrix_train = torch.stack([train_loss[k] for k in sorted(train_loss.keys())])
    loss_matrix_val = torch.stack([val_loss[k] for k in sorted(val_loss.keys())])
    utils.plottrainvalepoch(loss_matrix_train[:,-1],loss_matrix_val[:,-1], config['run_dir'])
    
if __name__ == "__main__":
    engineer.fire(main)
