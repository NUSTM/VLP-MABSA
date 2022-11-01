from datetime import datetime

import numpy as np
from torch.cuda.amp import autocast
import src.model.utils as utils
import src.eval_utils as eval_utils
# from src.utils import TaskType


def pretrain(task_list,
             epoch,
             model,
             train_loaders,
             optimizer_dict,
             device,
             args,
             logger=None,
             callback=None,
             log_interval=1,
             tb_writer=None,
             tb_interval=1,
             scaler=None):

    # assert len(task_list) == len(train_loaders)

    total_step = len(train_loaders[0])
    model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batchs in enumerate(zip(*train_loaders)):
        # Forward pass
        with autocast(enabled=args.amp):
            loss_all = []
            total_loss = 0
            for cnt, task in enumerate(task_list):
                batch = batchs[cnt]
                # print(batch.keys())
                if task == 'Sentiment':
                    loss, prelogits = model.forward(
                        task,
                        input_ids=batch['input_ids'].to(device),
                        image_features=list(
                            map(lambda x: x.to(device),
                                batch['image_features'])),
                        attention_mask=batch['attention_mask'].to(device),
                        senti_infos={
                            key: value.to(device)
                            for key, value in batch['Sentiment'].items()
                        })
                else:
                    loss = model.forward(
                        task,
                        input_ids=batch['input_ids'].to(device),
                        image_features=list(
                            map(lambda x: x.to(device),
                                batch['image_features'])),
                        attention_mask=batch['attention_mask'].to(device),
                        mlm_infos={
                            key: value.to(device)
                            for key, value in batch['MLM'].items()
                        } if 'MLM' in batch else None,
                        mrm_infos={
                            key: value
                            for key, value in batch['MRM'].items()
                        } if 'MRM' in batch else None,
                        senti_infos={
                            key: value.to(device)
                            for key, value in batch['Sentiment'].items()
                        } if 'Sentiment' in batch else None,
                        ANP_infos={
                            key: value.to(device)
                            for key, value in batch['ANP'].items()
                        } if 'ANP' in batch else None,
                        ANP_generate_infos={
                            key: value.to(device)
                            for key, value in batch['ANP_generate'].items()
                        } if 'ANP_generate' in batch else None,
                        ae_oe_infos={
                            key: value
                            for key, value in batch['AE_OE'].items()
                        } if 'AE_OE' in batch else None)

                # print(loss.dtype)
                loss_all.append(loss)
                optimizer_dict.zero_grad()

                loss.backward()
                optimizer_dict.step()

            for k, v in zip(task_list, loss_all):
                print(k + ':', v.item(), end=' ')
            print()
        # Backward and optimize

        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}]'.format(
                epoch + 1, args.epochs, i + 1, total_step))
            loss_text = ' '.join(
                [k + ':' + str(v.item()) for k, v in zip(task_list, loss_all)])
            logger.info(loss_text + '\n')


def fine_tune(epoch,
              model,
              train_loader,
              test_loader,
              metric,
              optimizer,
              device,
              args,
              logger=None,
              callback=None,
              log_interval=1,
              tb_writer=None,
              tb_interval=1,
              scaler=None):

    total_step = len(train_loader)
    model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batch in enumerate(train_loader):
        # Forward pass
        if args.task == 'twitter_ae':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_AE'].items()
            }
        elif args.task == 'twitter_sc':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_SC'].items()
            }
        else:
            aesc_infos = {key: value for key, value in batch['AESC'].items()}
        with autocast(enabled=args.amp):
            loss = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(
                    map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                aesc_infos=aesc_infos)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, args.epochs, i + 1, total_step, loss.item()))
        # Backward and optimize

        cur_step = i + 1 + epoch * total_step
        t_step = args.epochs * total_step
        liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
        utils.set_lr(optimizer, liner_warm_rate * args.lr)

        optimizer.zero_grad()

        loss.backward()
        utils.clip_gradient(optimizer, args.grad_clip)

        optimizer.step()
