import argparse
import math
import os
from datetime import datetime
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_metric
from transformers import (AdamW, AutoConfig, AutoTokenizer, SchedulerType,
                          get_scheduler)

from metrics import task_to_metric_name
from data_loader import task_to_benchmark, task_to_collator, task_to_load_fns, ws_task_to_load_fns
from data_loader.multitask_dataset import (MultitaskBatchSampler,
                                           MultitaskCollator, MultitaskDataset)
from models.modeling_multi_bert import MultitaskBertForClassification
from utils import add_result_to_csv, get_logger, setup_logging
from trainer import *

def setup_logging_logic():
    # set up log file
    log_dir = "./saved/logs/{}".format(datetime.now().strftime(r'%m%d_%H%M%S'))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    setup_logging(log_dir)

def load_ws_task_data(task_name, model_name_or_path, pad_to_max_length, max_length, batch_size, lf_idxes, ws_method, ws_params,
                    downsample_frac=1.0):
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    train_datasets, valid_dataset, test_dataset, num_lfs, num_labels = ws_task_to_load_fns[task_name](
            task_name=task_name, tokenizer = tokenizer,
            pad_to_max_length=pad_to_max_length, max_length=max_length, ws_method=ws_method, ws_params=ws_params)
    assert min(lf_idxes) >= 0 and max(lf_idxes) <= num_lfs

    rng = np.random.default_rng(int.from_bytes(task_name.encode("utf-8"), "little"))
    if 0 < downsample_frac < 1.0:
        for i, train_dataset in enumerate(train_datasets):
            tmp_len = len(train_dataset)
            if tmp_len > 100: # skip ws dataset with size less than 100
                downsample_len = int(tmp_len*downsample_frac) 
                indices = rng.choice(tmp_len, downsample_len, replace=False)
                train_datasets[i] = Subset(train_dataset, indices=indices)
    print("Data set lengths: " + " ".join([str(len(dataset)) for dataset in train_datasets]))

    train_dataloaders = [
        DataLoader(train_dataset, collate_fn=task_to_collator[task_name], batch_size=batch_size, shuffle=True) for train_dataset in train_datasets
        ]
    valid_dataloader = DataLoader(valid_dataset, collate_fn=task_to_collator[task_name], batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=task_to_collator[task_name], batch_size=batch_size)

    task_to_train_datasets = dict([
        ("{}_{}".format(task_name, i), train_datasets[i]) for i in lf_idxes
        ])
    task_to_train_dataloaders = dict([
        ("{}_{}".format(task_name, i), train_dataloaders[i]) for i in lf_idxes
        ])
    task_to_valid_dataloaders = dict([
        ("{}_{}".format(task_name, i), valid_dataloader) for i in [0] # + lf_idxes
        ])
    task_to_test_dataloaders = dict([
        ("{}_{}".format(task_name, i), test_dataloader) for i in [0] # + lf_idxes
        ])
    task_to_num_labels = dict([
        ("{}_{}".format(task_name, i), num_labels) for i in [0] + lf_idxes
        ])
    
    task_to_metrics = dict([
        ("{}_{}".format(task_name, i), load_metric(task_to_metric_name[task_name], task_name)) for i in [0] + lf_idxes
        ])

    multitask_train_dataset = MultitaskDataset(task_to_train_datasets)
    multitask_train_sampler = MultitaskBatchSampler(task_to_train_datasets, batch_size)
    multitask_train_collator = MultitaskCollator(task_to_collator)
    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        batch_sampler=multitask_train_sampler,
        collate_fn=multitask_train_collator.collator_fn,
    )
    return multitask_train_dataloader, task_to_train_dataloaders, task_to_valid_dataloaders, task_to_test_dataloaders, \
        task_to_num_labels, task_to_metrics, config, tokenizer 

def load_task_data(task_names, model_name_or_path, pad_to_max_length, max_length, batch_size, 
                    downsample_frac=1.0):
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    task_to_train_datasets = {}
    task_to_train_dataloaders = {}
    task_to_valid_dataloaders = {}
    task_to_test_dataloaders = {}
    task_to_num_labels = {}

    for task_name in task_names:
        train_dataset, valid_dataset, test_dataset, num_labels = task_to_load_fns[task_name](
            task_name=task_name, benchmark_name=task_to_benchmark[task_name],
            config = config, tokenizer = tokenizer,
            pad_to_max_length=pad_to_max_length, max_length=max_length)

        if 0 < downsample_frac < 1.0:
            tmp_len = len(train_dataset)
            if tmp_len > 100: # skip ws dataset with size less than 100
                rng = np.random.default_rng(int.from_bytes(task_name.encode("utf-8"), "little"))
                downsample_len = int(tmp_len*downsample_frac) 
                indices = rng.choice(tmp_len, downsample_len, replace=False)
                train_dataset = Subset(train_dataset, indices=indices)

        train_dataloader = DataLoader(train_dataset, collate_fn=task_to_collator[task_name], batch_size=batch_size)
        valid_dataloader = DataLoader(valid_dataset, collate_fn=task_to_collator[task_name], batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=task_to_collator[task_name], batch_size=batch_size) \
            if test_dataset is not None else None

        task_to_train_datasets[task_name] = train_dataset
        task_to_train_dataloaders[task_name] = train_dataloader
        task_to_valid_dataloaders[task_name] = valid_dataloader
        task_to_test_dataloaders[task_name] = test_dataloader
        task_to_num_labels[task_name] = num_labels

    print("Dataset lengths: " + " ".join([str(len(dataset)) for dataset in task_to_train_datasets.values()]))

    multitask_train_dataset = MultitaskDataset(task_to_train_datasets)
    multitask_train_sampler = MultitaskBatchSampler(task_to_train_datasets, batch_size)
    multitask_train_collator = MultitaskCollator(task_to_collator)
    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        batch_sampler=multitask_train_sampler,
        collate_fn=multitask_train_collator.collator_fn,
    )

    # get the metric function
    task_to_metrics = {}
    for task_name in task_names:
        task_to_metrics[task_name] = load_metric(task_to_metric_name[task_name], task_name)

    return multitask_train_dataloader, task_to_train_dataloaders, task_to_valid_dataloaders, task_to_test_dataloaders, \
        task_to_num_labels, task_to_metrics, config, tokenizer 

def main(args):
    setup_logging_logic()
    logger = get_logger("main")

    # load multitask data
    if args.use_ws_dataset:
        ws_params = {
            'lr': args.ws_lr,
            'l2': args.ws_l2,
            'n_epochs': args.ws_epochs,
        }
        multitask_train_dataloader, task_to_train_loaders, task_to_valid_loaders, task_to_test_loaders, \
        task_to_num_labels, task_to_metrics, transformer_config, transformer_tokenizer = \
            load_ws_task_data(args.ws_task_name, args.model_name_or_path, 
            args.pad_to_max_length, args.max_length, args.batch_size, args.lf_idxes, args.ws_method, ws_params,
            args.downsample_frac)
    else:
        multitask_train_dataloader, task_to_train_loaders, task_to_valid_loaders, task_to_test_loaders, \
        task_to_num_labels, task_to_metrics, transformer_config, transformer_tokenizer = \
            load_task_data(args.task_names, args.model_name_or_path, 
            args.pad_to_max_length, args.max_length, args.batch_size,
            args.downsample_frac)
    
    task_names = list(task_to_train_loaders.keys())

    # initialize model
    multitask_model = MultitaskBertForClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config = transformer_config,
        tasks = list(task_to_num_labels.keys()),
        num_labels_list = list(task_to_num_labels.values()),
        use_one_predhead = args.use_one_predhead
    )

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    multitask_model.to(device)
    valid_metrics = {}; test_metrics = {}
    for run in range(args.runs):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in multitask_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in multitask_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

        # Scheduler and math around the number of training steps.
        trainer_config = { 
            "num_train_epochs": args.epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_train_steps": args.max_train_steps,
            "num_warmup_steps": args.num_warmup_steps,

            "save_dir": args.save_dir,
            "save_period": args.save_period,
            "verbosity": args.verbosity,
            
            "monitor": " ".join([args.monitor_mode, args.monitor_metric]),
            "early_stop": args.early_stop
        }
        num_update_steps_per_epoch = math.ceil(len(multitask_train_dataloader) / trainer_config["gradient_accumulation_steps"])
        if trainer_config["max_train_steps"] == -1:
            trainer_config["max_train_steps"] = trainer_config["num_train_epochs"] * num_update_steps_per_epoch
        else:
            trainer_config["num_train_epochs"] = math.ceil(trainer_config["max_train_steps"] / num_update_steps_per_epoch)
        
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=trainer_config["num_warmup_steps"]*num_update_steps_per_epoch,
            num_training_steps=trainer_config["max_train_steps"],
        )

        lf_idxes = [str(idx) for idx in args.lf_idxes]

        if args.train_tawt: 
            checkpoint_dir = os.path.join(
                "saved", 
                args.save_name + "_" + args.ws_task_name + "_".join(lf_idxes) + "_tawt" if len(lf_idxes) < 70 else \
                    args.save_name + "_" + args.ws_task_name + "_".join(lf_idxes[:70]) + "_tawt"
                )
            trainer = TAWTrainer(multitask_model, optimizer, lr_scheduler, trainer_config, device, task_to_metrics, 
                                    multitask_train_data_loader=multitask_train_dataloader,
                                    train_data_loaders = task_to_train_loaders,
                                    valid_data_loaders = task_to_valid_loaders,
                                    test_data_loaders = task_to_test_loaders if args.use_ws_dataset else None, # stop using test datasets (label not available)
                                    task_to_num_labels = task_to_num_labels,
                                    checkpoint_dir = checkpoint_dir,
                                    target_task = args.tawt_target,
                                    weight_lr = args.tawt_lr, 
                                    collect_gradient_step = args.tawt_steps, 
                                    use_target_train = args.tawt_use_target_train)
        elif args.train_autol: 
            checkpoint_dir = os.path.join(
                "saved", 
                args.save_name + "_" + args.ws_task_name + "_".join(lf_idxes) + "_tawt" if len(lf_idxes) < 70 else \
                    args.save_name + "_" + args.ws_task_name + "_".join(lf_idxes[:70]) + "_tawt"
                )
            trainer = AutoLTrainer(multitask_model, optimizer, lr_scheduler, trainer_config, device, task_to_metrics, 
                                    multitask_train_data_loader=multitask_train_dataloader,
                                    train_data_loaders = task_to_train_loaders,
                                    valid_data_loaders = task_to_valid_loaders,
                                    test_data_loaders = task_to_test_loaders if args.use_ws_dataset else None, # stop using test datasets (label not available)
                                    task_to_num_labels = task_to_num_labels,
                                    checkpoint_dir = checkpoint_dir,
                                    target_task = args.autol_target,
                                    weight_lr = args.autol_lr,
                                    update_weight_step = args.autol_step)
        elif args.train_attittud:
            checkpoint_dir = os.path.join(
                "saved", 
                args.save_name + "_" + args.ws_task_name + "_".join(lf_idxes) + "_attittud" if len(lf_idxes) < 70 else \
                    args.save_name + "_" + args.ws_task_name + "_".join(lf_idxes[:70]) + "_attittud"
                )
            trainer = AttitudeTrainer(multitask_model, optimizer, lr_scheduler, trainer_config, device, task_to_metrics, 
                                    multitask_train_data_loader=multitask_train_dataloader,
                                    train_data_loaders = task_to_train_loaders,
                                    valid_data_loaders = task_to_valid_loaders,
                                    test_data_loaders = task_to_test_loaders if args.use_ws_dataset else None, # stop using test datasets (label not available)
                                    task_to_num_labels = task_to_num_labels,
                                    checkpoint_dir = checkpoint_dir,
                                    target_task = args.attittud_target,
                                    collect_gradient_step = args.attittud_steps, 
                                    use_target_train = args.attittud_use_target_train,
                                    eta_tilde = args.eta_tilde, 
                                    eta_pos   = args.eta_pos, 
                                    eta_neg   = -args.eta_neg)
        else:
            checkpoint_dir = os.path.join(
                "saved", 
                args.save_name + "_" + args.ws_task_name + "_".join(lf_idxes) if len(lf_idxes) < 70 else \
                    args.save_name + "_" + args.ws_task_name + "_".join(lf_idxes[:70])
                )
            trainer = MultitaskTrainer(multitask_model, optimizer, lr_scheduler, trainer_config, device, task_to_metrics, 
                                    multitask_train_data_loader=multitask_train_dataloader,
                                    train_data_loaders = task_to_train_loaders,
                                    valid_data_loaders = task_to_valid_loaders,
                                    test_data_loaders = task_to_test_loaders if args.use_ws_dataset else None, # stop using test datasets (label not available)
                                    task_to_num_labels = task_to_num_labels,
                                    checkpoint_dir = checkpoint_dir,
                                    eval_margin = True)
        if not (len(list(os.listdir(checkpoint_dir)))!=0 and args.skip_pretrain):
            start_time = time.time()
            trainer.train()
            end_time = time.time()
            run_time = end_time - start_time
            logger.info("Running time: {}".format(run_time))
        if args.train_finetune:
            valid_log = {}; test_log = {}
            finetune_tasks = task_names if not args.finetune_target else args.target_tasks
            for task in finetune_tasks:
                finetune_checkpoint_dir = checkpoint_dir + "_finetune_{}".format(task)
                trainer.load_best_model(strict=False)

                # reinitiate optimizer & lr_scheduler
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in multitask_model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [p for n, p in multitask_model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
                optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
                # change trainer config to finetune config
                trainer_config['num_train_epochs'] = args.finetune_epochs
                trainer_config["max_train_steps"] = args.max_train_steps
                num_update_steps_per_epoch = math.ceil(len(task_to_train_loaders[task]) / trainer_config["gradient_accumulation_steps"])
                if trainer_config["max_train_steps"] == -1:
                    trainer_config["max_train_steps"] = trainer_config["num_train_epochs"] * num_update_steps_per_epoch
                else:
                    trainer_config["num_train_epochs"] = math.ceil(trainer_config["max_train_steps"] / num_update_steps_per_epoch)
                lr_scheduler = get_scheduler(
                    name=args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=trainer_config["num_warmup_steps"]*num_update_steps_per_epoch,
                    num_training_steps=trainer_config["max_train_steps"],
                )

                # create fine-tune trainer
                finetune_trainer = MultitaskTrainer(multitask_model, optimizer, lr_scheduler, trainer_config, device, 
                                    task_to_metrics = {task: task_to_metrics[task]}, 
                                    multitask_train_data_loader=multitask_train_dataloader,
                                    train_data_loaders = {task: task_to_train_loaders[task]},
                                    valid_data_loaders = {task: task_to_valid_loaders[task]},
                                    test_data_loaders = task_to_test_loaders if args.use_ws_dataset else None, # stop using test datasets (label not available)
                                    task_to_num_labels = {task: task_to_num_labels[task]},
                                    checkpoint_dir = finetune_checkpoint_dir)
                finetune_trainer.train()

                task_valid_log = finetune_trainer.test(use_valid = True)
                task_test_log = finetune_trainer.test(use_valid = False)
                valid_log.update(task_valid_log)
                test_log.update(task_test_log)
            if args.finetune_target:
                mtl_valid_log = trainer.test(use_valid=True)
                mtl_test_log = trainer.test(use_valid=False) # update log of other tasks
                for key, val in mtl_valid_log.items():
                    if (key == 'loss') or (key not in valid_log):
                        valid_log[key] = val
                for key, val in mtl_test_log.items():
                    if (key == 'loss') or (key not in test_log):
                        test_log[key] = val
        else:
            valid_log = trainer.test(use_valid = True)
            test_log = trainer.test(use_valid = False)
        
        for key, val in valid_log.items():
            if key in valid_metrics:
                valid_metrics[key].append(val)
            else:
                valid_metrics[key] = [val, ]
        
        for key, val in test_log.items():
            if key in test_metrics:
                test_metrics[key].append(val)
            else:
                test_metrics[key] = [val, ]

        # re-initialize model
        multitask_model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model_epoch_0.pth"))['state_dict']
        )
    
    # print training results
    for key, vals in test_metrics.items():
        logger.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(vals), np.std(vals)))

    # save results into .csv
    file_dir = os.path.join("./results/", args.save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    for task_name in [f"{args.ws_task_name}_0"]:
        # save validation results
        result_datapoint = {
            "Task": task_name, 
            "Trained on": task_names,
        }
        for key, vals in valid_metrics.items():
            if task_name in key:
                result_datapoint[key] = np.mean(vals)
                result_datapoint[key+"_std"] = np.std(vals)
        file_name = os.path.join(file_dir, "{}_{}_valid.csv".format(args.save_name, task_name))
        add_result_to_csv(result_datapoint, file_name)
        # save test results
        result_datapoint = {
            "Task": task_name, 
            "Trained on": task_names,
        }
        for key, vals in test_metrics.items():
            if task_name in key:
                result_datapoint[key] = np.mean(vals)
                result_datapoint[key+"_std"] = np.std(vals)
        file_name = os.path.join(file_dir, "{}_{}_test.csv".format(args.save_name, task_name))
        add_result_to_csv(result_datapoint, file_name)

def add_trainer_arguments(parser):
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=-1)
    parser.add_argument("--num_warmup_steps", type=int, default=0)

    parser.add_argument("--save_dir", type=str, default="./saved/test/")
    parser.add_argument("--save_period", type=int, default=1)
    parser.add_argument("--verbosity", type=int, default=2)

    parser.add_argument("--monitor_mode", type=str, default='off', choices=['min', 'max', 'off'])
    parser.add_argument("--monitor_metric", type=str, default='val_loss')
    parser.add_argument("--early_stop", type=int, default=10)
    return parser

def add_dataloader_arguments(parser):
    parser.add_argument('--pad_to_max_length', default=True)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser

def add_finetune_arguments(parser):
    parser.add_argument("--train_finetune", action="store_true")
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--finetune_epochs", type=int, default=5)

    parser.add_argument("--finetune_target", action="store_true")
    parser.add_argument("--target_tasks", nargs='+', default=['mrpc'])
    return parser

def add_ws_arguments(parser):
    parser.add_argument("--use_ws_dataset", action="store_true")
    parser.add_argument("--use_one_predhead", action="store_true")
    parser.add_argument("--ws_task_name", type=str, default="youtube")
    parser.add_argument("--lf_idxes", nargs='+', type=int)

    parser.add_argument("--ws_method", type=str, default="none")
    # Weak supervision method parameters
    parser.add_argument("--ws_lr", type=float, default=1e-4)
    parser.add_argument("--ws_l2", type=float, default=1e-1)
    parser.add_argument("--ws_epochs", type=int, default=100)
    # Weak supervision method parameters
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_names", nargs='+', default=['mrpc'])
    parser.add_argument("--model_name_or_path", type=str, default="prajjwal1/bert-mini")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--save_name", type=str, default="test")
    parser.add_argument("--downsample_frac", type=float, default=1.0)

    # TAWT arguments
    parser.add_argument("--train_tawt", action="store_true")
    parser.add_argument("--tawt_lr", type=float, default=1)
    parser.add_argument("--tawt_steps", type=int, default=10)
    parser.add_argument("--tawt_use_target_train", action="store_true")
    parser.add_argument("--tawt_target", type=str, default="youtube_0")

    # Auto-Lambda 
    parser.add_argument("--train_autol", action="store_true")
    parser.add_argument("--autol_lr", type=float, default=1e-4)
    parser.add_argument("--autol_step", type=int, default=100)
    parser.add_argument("--autol_target", type=str, default="youtube_0")

    # ATTITTUD arguments
    parser.add_argument("--train_attittud", action="store_true")
    parser.add_argument("--attittud_steps", type=int, default=2)
    parser.add_argument("--attittud_use_target_train", action="store_true")
    parser.add_argument("--attittud_target", type=str, default="youtube_0")
    parser.add_argument("--eta_tilde", type=float, default=1)
    parser.add_argument("--eta_pos", type=float, default=1)
    parser.add_argument("--eta_neg", type=float, default=0)

    parser = add_trainer_arguments(parser)
    parser = add_dataloader_arguments(parser)
    parser = add_finetune_arguments(parser)
    parser = add_ws_arguments(parser)
    args = parser.parse_args()

    main(args)
