import torch
import torch.nn as nn


def eval(args, model, loader, metric, device):
    model.eval()
    for i, batch in enumerate(loader):
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
        predict = model.predict(
            input_ids=batch['input_ids'].to(device),
            image_features=list(
                map(lambda x: x.to(device), batch['image_features'])),
            attention_mask=batch['attention_mask'].to(device),
            aesc_infos=aesc_infos)

        metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))
        # break

    res = metric.get_metric()
    model.train()
    return res
