from transformers.generation_utils import top_k_top_p_filtering
import torch
from torch.nn import functional as F


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            # print(param.shape)
            if param.grad == None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)


def liner_warmup(cur_step, t_step, warmup):
    progress = cur_step / t_step
    if progress < warmup:
        return progress / warmup
    return max((progress - 1.) / (warmup - 1.), 0.)


def sample_sentence(model,
                    input_ids,
                    image_features,
                    attention_mask,
                    tokenizer,
                    top_k=50,
                    top_p=1.0,
                    max_length=20):
    batch_size = input_ids.shape[0]
    encoder = model.get_encoder()
    encoder_outputs = encoder(input_ids,
                              image_features,
                              attention_mask=attention_mask)

    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)
    logprobs = []

    decoder_input_ids = input_ids.new(batch_size,
                                      1).fill_(tokenizer.bos_token_id)

    cur_len = 1
    while cur_len < max_length:
        model_inputs = {
            "input_ids": None,
            "decoder_input_ids": decoder_input_ids,
            "image_features": image_features,
            "attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs,
            "use_cache": False
        }

        outputs = model(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits,
                                                  top_k=top_k,
                                                  top_p=top_p)

        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1),
                                       num_samples=1).squeeze(1)

        _scores = F.log_softmax(next_token_logits, dim=-1)
        _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))
        logprobs.append(_scores)

        tokens_to_add = next_token * unfinished_sents + (
            tokenizer.pad_token_id) * (1 - unfinished_sents)
        decoder_input_ids = torch.cat(
            [decoder_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1

        eos_in_sents = tokens_to_add == tokenizer.eos_token_id
        # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
        is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(
            eos_in_sents.long()).bool()
        sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos,
                                  cur_len)
        # unfinished_sents is set to zero if eos in sentence
        unfinished_sents.mul_((~eos_in_sents).long())

        if unfinished_sents.max() == 0:
            break

    logprobs = torch.cat(logprobs, dim=1)
    for i in range(batch_size):
        logprobs[i, sent_lengths[i] - 1:] = 0

    sum_logprobs = logprobs.sum(dim=1)

    return decoder_input_ids, sum_logprobs.unsqueeze(1)
