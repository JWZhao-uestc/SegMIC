import sys
from pathlib import Path
import yaml
import json
import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
#import click
import argparse

from segm import config

from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params

from timm.utils import NativeScaler
from contextlib import suppress

from segm.engine import train_one_epoch, evaluate
import warnings
warnings.filterwarnings("ignore")

def main(args):
    ####param
    freeze_mask_embed = args.freeze_mask_embed
    freeze_image_embed = args.freeze_image_embed
    freeze_post_process = args.freeze_post_process
    painter_depth = args.painter_depth
    data_root = args.data_root
    db_name = args.db_name
    log_dir = args.log_dir
    dataset = args.dataset
    im_size = args.im_size
    crop_size = args.crop_size
    window_size = args.window_size
    window_stride = args.window_stride
    backbone = args.backbone
    decoder = args.decoder
    optimizer = args.optimizer
    scheduler = args.scheduler
    weight_decay = args.weight_decay
    dropout = args.dropout
    drop_path = args.drop_path
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    normalization = args.normalization
    eval_freq = args.eval_freq
    amp = args.amp
    resume = args.resume
    min_lr = args.min_lr
    # start distributed mode
    # distributed.init_process()

    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    model_cfg["painter_depth"] = painter_depth

    # decoder_cfg["name"] = decoder
    # decoder_cfg["n_cls"] = n_cls
    # model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]
    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    batch_size = world_batch_size
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            data_root=data_root,
            db_name=db_name,
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=12, ########
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=min_lr,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    #log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_dir / "all_save_checkpoint_with_genloss_different_encoder_v2.pth"#"all_save_checkpoint_wo_centerloss.pth" #"checkpoint.pth" #"save_checkpoint_path_with_center_loss.pth" #"save_checkpoint_path.pth" #"checkpoint.pth" !!!!!!!!!!!!

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)
    n_cls = train_loader.unwrapped.n_cls
    
    decoder_cfg["name"] = decoder
    decoder_cfg["n_cls"] = n_cls
    model_cfg["decoder"] = decoder_cfg

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    model = create_segmenter(net_kwargs)
    IDS = [0,1]
    if args.n_gpus:
        print("***")
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=IDS, output_device=0)
    else:
        model.cuda()
    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    
    if freeze_mask_embed:
        total_freeze_paras = 0
        for name, para in model.named_parameters():
            k = name
            if 'mask_downscaling' in k:
                #print(k) 
                total_freeze_paras += 1
                para.requires_grad = False
            else:
                #print(k)
                para.requires_grad = True
        print("!!!!!!!!!!!!Stage2: there are total %d parameters are freezing during training" % (total_freeze_paras))
        parameters_to_optimze = filter(lambda p: p.requires_grad, model.parameters())
        #optimizer = create_optimizer(opt_args, parameters_to_optimze)
        optimizer = torch.optim.SGD(parameters_to_optimze, lr=opt_args.lr, momentum=opt_args.momentum)
        # assert checkpoint_path.name != 'checkpoint.pth'
    if freeze_image_embed:
        total_freeze_paras = 0
        for name, para in model.named_parameters():
            k = name
            if 'encoder' in k and 'painter' not in k:
                #print(k) 
                total_freeze_paras += 1
                para.requires_grad = False
            else:
                #print(k)
                para.requires_grad = True
        print("!!!!!!!!!!!!Stage3: there are total %d parameters are freezing during training" % (total_freeze_paras))
        parameters_to_optimze = filter(lambda p: p.requires_grad, model.parameters())
        assert total_freeze_paras == 156
        #optimizer = create_optimizer(opt_args, parameters_to_optimze)
        optimizer = torch.optim.SGD(parameters_to_optimze, lr=opt_args.lr, momentum=opt_args.momentum)
        #import pdb
        #pdb.set_trace()
        # assert checkpoint_path.name != 'checkpoint.pth'
    if freeze_post_process:
        total_freeze_paras = 0
        for name, para in model.named_parameters():
            k = name
            if 'painter' in k or 'decoder' in k:
                print(k) 
                total_freeze_paras += 1
                para.requires_grad = False
            else:
                #print(k)
                para.requires_grad = True
        print("!!!!!!!!!!!!Stage4: there are total %d parameters are freezing during training" % (total_freeze_paras))
        parameters_to_optimze = filter(lambda p: p.requires_grad, model.parameters())
        assert total_freeze_paras == 128
        #optimizer = create_optimizer(opt_args, parameters_to_optimze)
        optimizer = torch.optim.SGD(parameters_to_optimze, lr=opt_args.lr, momentum=opt_args.momentum)
    if not (freeze_mask_embed or freeze_mask_embed):
        # stage1: optimize all parameters
        print("!!!!!!!!!!!!Stage1: train whole model")
        optimizer = create_optimizer(opt_args, model)
        print(checkpoint_path.name)
        # assert checkpoint_path.name == 'checkpoint.pth'
    
    lr_scheduler = create_scheduler(opt_args, optimizer)
    num_iterations = 0
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    # print(resume, checkpoint_path.exists())
    # exit()
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_name = Path(checkpoint_path)
        if checkpoint_name.name == "checkpoint.pth":
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            #print("**********", checkpoint_name.name)
            new_state_dict = {}
            state_dict = checkpoint['model']
            for k, v in state_dict.items():
                if 'decoder.cls_emb' not in k and 'decoder.mask_norm' not in k:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        elif checkpoint_name.name == "all_save_checkpoint_with_genloss_bone_different_encoder_v2_freeze_ab_mask75_painterx6.pth": #"save_checkpoint_path_lung70-96.pth":
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            new_state_dict = {}
            state_dict = checkpoint['model']
            for k, v in state_dict.items():
                if 'decoder' not in k and 'painter' not in k:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        
        else:
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            model.load_state_dict(checkpoint["model"], strict=False)
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # if loss_scaler and "loss_scaler" in checkpoint:
        #     loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1
        variant["algorithm_kwargs"]["start_epoch"] = 0
    # save config
    variant_str = yaml.dump(variant)
    print(f"Configuration:\n{variant_str}")
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    #print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    #print(f"Add Encoder parameters: {num_params(model_without_ddp.add_encoder)}")
    #print(f"Add Painter Encoder parameters: {num_params(model_without_ddp.add_cor_encoder)}")
    #print(f"Add Segmenter Encoder parameters: {num_params(model_without_ddp.add_seg_encoder)}")
    
    # print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")
    dice_max = 0.0 #0.40
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            crop_size,
            IDS
        )
        # save checkpoint
        snapshot = dict(
            model=model_without_ddp.state_dict(),
            optimizer=optimizer.state_dict(),
            n_cls=model_without_ddp.n_cls,
            lr_scheduler=lr_scheduler.state_dict(),
        )
        if loss_scaler is not None:
            snapshot["loss_scaler"] = loss_scaler.state_dict()
        snapshot["epoch"] = epoch
        
        # evaluate
        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            dice = evaluate(
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast
            )

        #############Save
        save_checkpoint_path = log_dir / "all_save_checkpoint_with_genloss_different_encoder_v2_add_bone_v2.pth"
        if dice > dice_max:
            print('save_checkpoint_path', str(save_checkpoint_path))
            torch.save(snapshot, save_checkpoint_path)
            if epoch >= 3:
               dice_max = dice
        
        # log stats
        # train_stats = {
        #     k: meter.global_avg for k, meter in train_logger.meters.items()
        # }
        # val_stats = {}
        # if eval_epoch:
        #     val_stats = {
        #         k: meter.global_avg for k, meter in eval_logger.meters.items()
        #     }

        # log_stats = {
        #     **{f"train_{k}": v for k, v in train_stats.items()},
        #     **{f"val_{k}": v for k, v in val_stats.items()},
        #     "epoch": epoch,
        #     "num_updates": (epoch + 1) * len(train_loader),
        # }

        # with open(log_dir / "log.txt", "a") as f:
        #     f.write(json.dumps(log_stats) + "\n")

    # distributed.barrier()
    # distributed.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--freeze_mask_embed", action="store_true")
    parser.add_argument("--freeze_image_embed", action="store_true")
    parser.add_argument("--freeze_post_process", action="store_true")
    parser.add_argument("--painter_depth", default=4, type=int)
    parser.add_argument("--log-dir", type=str, default='seg_patch16_384_bce',help="logging directory")
    parser.add_argument("--data_root", type=str, default='./Raw_data', help="logging directory")
    parser.add_argument("--db_name", type=str, default='all', help="which dataset to train")
    parser.add_argument("--dataset", type=str, default='dis5k')
    parser.add_argument("--im-size", default=None, type=int, help="dataset resize size")
    parser.add_argument("--crop-size", default=None, type=int)
    parser.add_argument("--window-size", default=None, type=int)
    parser.add_argument("--window-stride", default=None, type=int)
    parser.add_argument("--backbone", default="vit_small_patch16_384", type=str)
    parser.add_argument("--decoder", default="mask_transformer", type=str)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--scheduler", default="polynomial", type=str)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--drop-path", default=0.1, type=float)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=512, type=int)
    parser.add_argument("-lr", "--learning-rate", default=None, type=float)
    parser.add_argument("--min-lr", default=1e-6, type=float)
    parser.add_argument("--normalization", default=None, type=str)
    parser.add_argument("--eval-freq", default=1, type=int)
    parser.add_argument("--amp", default=False, type=bool)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_gpus", default=False, type=bool)
    args = parser.parse_args()
    main(args)