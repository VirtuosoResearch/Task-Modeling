from utils import prepare_inputs
from trainer.base_trainer import MultitaskTrainer
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from utils.autol import update_task_weights

class AutoLTrainer(MultitaskTrainer):
    
    def __init__(self, model, optimizer, lr_scheduler, config, device, task_to_metrics, multitask_train_data_loader, 
        train_data_loaders, valid_data_loaders=None, test_data_loaders=None, 
        task_to_num_labels=None, checkpoint_dir=None, criterion=None,
        target_task = None, weight_lr=1, update_weight_step=100):
        super().__init__(model, optimizer, lr_scheduler, config, device, task_to_metrics, 
            multitask_train_data_loader, train_data_loaders, valid_data_loaders, test_data_loaders, 
            task_to_num_labels, checkpoint_dir, criterion)
        
        self.model_ = copy.deepcopy(model)
        
        self.task_list = list(train_data_loaders.keys())
        self.num_tasks = len(self.task_list)
        self.task_to_index = dict([(task, i) for i, task in enumerate(self.task_list)])

        self.target_task = target_task
        self.task_weights = torch.tensor([1/self.num_tasks] * self.num_tasks, device=self.device, requires_grad=True)
        self.meta_optimizer = optim.Adam([self.task_weights], lr=weight_lr)
        self.update_weight_step = update_weight_step

        # self.collect_gradient_step = collect_gradient_step
        # self.target_data_loader = self.train_data_loaders[target_task] if use_target_train else self.valid_data_loaders[target_task]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for step, batch in enumerate(self.multitask_train_data_loader):
            task_name = batch['task_name']
            batch = batch['data']
            
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch, task_name = task_name)
            loss = outputs.loss
            loss = loss / self.cfg_trainer["gradient_accumulation_steps"]

            ''' Apply weighted training'''
            loss = loss*self.task_weights[self.task_to_index[task_name]]*self.num_tasks
            ''' Apply weighted training'''

            loss.backward()
            if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(self.multitask_train_data_loader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.completed_steps += 1

            # update training metrics
            self.train_metrics.update('loss', loss.item())
            
            predictions = outputs.predictions
            labels = batch["labels"]
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1)
            self.task_to_metrics[task_name].add_batch(
                predictions=predictions,
                references=labels,
            )
            
            if self.completed_steps > self.cfg_trainer["max_train_steps"]:
                break

            '''
            Update task weights
            '''
            if (self.completed_steps) % self.update_weight_step == 0:
                self.meta_optimizer.zero_grad()
                update_task_weights(self.train_data_loaders, self.valid_data_loaders, self.model, self.model_,
                                    self.lr_scheduler.get_last_lr()[0], self.optimizer, self.task_weights, self.task_to_index, [self.target_task], self.device)
                self.meta_optimizer.step()
                self.logger.info(self.task_weights)


        log = self.train_metrics.result()
        for task_name in self.train_data_loaders.keys():
            metric = self.task_to_metrics[task_name]
            task_metrics = metric.compute()
            task_metrics = {f'{task_name}_{key}': val for key, val in task_metrics.items()}
            log.update(**task_metrics)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log