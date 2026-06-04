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
    wlr = optimizer_config.pop("wlr")
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


    # optimizer = engineer.load_module(optimizer_config.pop("module"))(
    #         get_param_groups(model,loss, wlr), **optimizer_config
    #         )
    
#    print("loss parameters", loss.named_parameters())
    # param_to_name = {}

    # for name, p in model.named_parameters():
    #     print(name)
    #     param_to_name[id(p)] = f"model.{name}"

    # for name, p in loss.named_parameters():
    #     print(name)
    #     param_to_name[id(p)] = f"loss.{name}"

    # for group in optimizer.param_groups:
    #     for p in group["params"]:
    #         print(param_to_name.get(id(p), "UNKNOWN"))

#    exit()        

#     name_to_param = {}

#     for name, param in model.named_parameters():
#         name_to_param[name] = param

#     for name, param in loss.named_parameters():
#         name_to_param[name] = param

# #    exit()
# # inspect optimizer groups
#     for group_idx, group in enumerate(optimizer.param_groups):
#         l = group.get("lr", None)
#         wd = group.get("weight_decay", None)

#         for p in group["params"]:
#             for name, param in name_to_param.items():
#                 if p is param:   # identity check (correct)
#                     print(f"{name}: lr = {l} w = {wd} (group {group_idx})")


#    exit()    
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


# def get_param_groups(model,loss, losswlr):
#     para1 = []
#     para2 = []

#     for name, param in model.named_parameters():
#         if not param.requires_grad :
#             continue

#         # Biases and normalization weights
# #        if name in ["lossfn.a_w1", "lossfn.z_raw"] :
#         if name in ["lossfn.w1", "lossfn.w2"]:
# #        if name in ["lossfn.log_w1_ud", "lossfn.log_w2_ud"]:
#             para1.append(param)
#         else:
#             para2.append(param) 

#     return [
#         {"params": para2},
#         {"params": para1, "lr": losswlr, "weight_decay": 0.0},
#     ]

def get_param_groups(model, loss, losswlr):
    """
    Two param groups:
    - model params (default optimizer settings)
    - loss params (custom hyperparameters)

    Args:
        model: nn.Module
        loss: nn.Module
        loss_hparams: dict (e.g. {"lr": 1e-4, "weight_decay": 0.0})

    Returns:
        list of param groups
    """

    model_params = [p for p in model.parameters() if p.requires_grad]
    loss_params  = [p for p in loss.parameters() if p.requires_grad]

    param_groups = [
        {"params": model_params},
    ]

    if loss_params:  # only add if non-empty
        param_groups.append({
            "params": loss_params,
            "lr": losswlr, "weight_decay": 0.0
        })

    return param_groups

if __name__ == "__main__":
    engineer.fire(main)
