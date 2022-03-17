# Based on transformers.modeling_bart

from typing import Optional, Tuple
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch
import torch.nn.functional as F
from torch import nn
from src.model.modeling_bart import (PretrainedBartModel, BartEncoder,
                                     BartDecoder, BartModel,
                                     BartClassificationHead,
                                     _make_linear_from_emb,
                                     _prepare_bart_decoder_inputs)

from transformers import BartTokenizer

from src.model.config import MultiModalBartConfig
from src.model.mixins import GenerationMixin, FromPretrainedMixin
from src.model.modules import MultiModalBartEncoder, MultiModalBartDecoder_span, MultiModalBartDecoder_MLM, MultiModalBartDecoder_sentiment, Span_loss, MultiModalBartDecoder_MRM, MultiModalBartDecoder_ANP_generate

# This is based on transformers.BartModel
# The modifications are:
# - BartConfig -> MultiModalBartConfig
# - BartEncoder -> MultiModalBartEncoder
# - added image_features in forward


# def generate_span_mask(spans):
#     max_len = max([len(x) for x in spans])
#     mask = torch.ones(())
class MultiModalBartModelForPretrain(FromPretrainedMixin, PretrainedBartModel):
    def build_model(self,
                    args,
                    bart_model,
                    tokenizer,
                    label_ids,
                    config,
                    decoder_type=None,
                    copy_gate=False,
                    use_encoder_mlp=False,
                    use_recur_pos=False,
                    tag_first=False):
        if args.bart_init:
            model = BartModel.from_pretrained(bart_model)
            num_tokens, _ = model.encoder.embed_tokens.weight.shape

            model.resize_token_embeddings(
                len(tokenizer.unique_no_split_tokens) + num_tokens)
            encoder = model.encoder
            decoder = model.decoder

            padding_idx = config.pad_token_id
            encoder.embed_tokens.padding_idx = padding_idx

            _tokenizer = BartTokenizer.from_pretrained(bart_model)

            for token in tokenizer.unique_no_split_tokens:
                if token[:2] == '<<':  # 特殊字符
                    index = tokenizer.convert_tokens_to_ids(
                        tokenizer._base_tokenizer.tokenize(token))
                    if len(index) > 1:
                        raise RuntimeError(f"{token} wrong split")
                    else:
                        index = index[0]
                    assert index >= num_tokens, (index, num_tokens, token)
                    indexes = _tokenizer.convert_tokens_to_ids(
                        _tokenizer.tokenize(token[2:-2]))
                    embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                    for i in indexes[1:]:
                        embed += model.decoder.embed_tokens.weight.data[i]
                    embed /= len(indexes)
                    model.decoder.embed_tokens.weight.data[index] = embed
        else:
            raise RuntimeError("error init!!!!!!!")

        multimodal_encoder = MultiModalBartEncoder(config, encoder,
                                                   tokenizer.img_feat_id,
                                                   tokenizer.cls_token_id)
        return (multimodal_encoder, decoder)

    def __init__(self, config: MultiModalBartConfig, bart_model, tokenizer,
                 label_ids, senti_ids, args):
        super().__init__(config)
        self.config = config
        label_ids = sorted(label_ids)
        multimodal_encoder, share_decoder = self.build_model(
            args, bart_model, tokenizer, label_ids, config)
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        self.causal_mask = causal_mask.triu(diagonal=1)
        self.encoder = multimodal_encoder
        self.mlm_decoder = MultiModalBartDecoder_MLM(self.config,
                                                     share_decoder)
        self.mrm_decoder = MultiModalBartDecoder_MRM(self.config,
                                                     share_decoder,
                                                     self.causal_mask, args)
        self.span_decoder = MultiModalBartDecoder_span(self.config, tokenizer,
                                                       share_decoder,
                                                       tokenizer.pad_token_id,
                                                       label_ids,
                                                       self.causal_mask)
        self.span_loss_fct = Span_loss()
        self.anp_generate_decoder = MultiModalBartDecoder_ANP_generate(
            self.config, share_decoder)
        self.senti_decoder = MultiModalBartDecoder_sentiment(
            self.config, share_decoder, senti_ids)

    def prepare_state(self,
                      input_ids,
                      image_features,
                      attention_mask=None,
                      first=None):
        dict = self.encoder(input_ids=input_ids,
                            image_features=image_features,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        encoder_mask = attention_mask
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, input_ids[:, 38:],
                          first, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(
            self,
            task_type,
            input_ids,
            image_features,
            attention_mask=None,
            mlm_infos=None,
            mrm_infos=None,
            senti_infos=None,
            ANP_infos=None,
            ANP_generate_infos=None,
            ae_infos=None,
            oe_infos=None,
            ae_oe_infos=None,
            encoder_outputs: Optional[Tuple] = None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                image_features=image_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        assert isinstance(encoder_outputs, tuple)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if task_type == 'MLM':
            labels, decoder_input_ids, decoder_attention_mask = [
                mlm_infos['mlm_labels'], mlm_infos['mlm_decoder_input_ids'],
                mlm_infos['mlm_decoder_attention_mask']
            ]
            loss = self.mlm_decoder(labels, input_ids, encoder_outputs[0],
                                    attention_mask, decoder_input_ids,
                                    decoder_attention_mask)
        elif task_type == 'MRM':
            mrm_labels, mrm_masks, decoder_input_ids, decoder_attention_mask = [
                mrm_infos['mrm_labels'],
                mrm_infos['mrm_masks'].to(input_ids.device),
                mrm_infos['mrm_decoder_input_ids'].to(input_ids.device),
                mrm_infos['mrm_decoder_attention_mask'].to(input_ids.device)
            ]
            loss = self.mrm_decoder(mrm_labels, mrm_masks, encoder_outputs[0],
                                    attention_mask, decoder_input_ids,
                                    decoder_attention_mask)
        elif task_type == 'Sentiment':
            senti_labels, decoder_input_ids, decoder_attention_mask = [
                senti_infos['senti_labels'],
                senti_infos['senti_decoder_input_ids'],
                senti_infos['senti_decoder_attention_mask']
            ]
            loss, predict_senti = self.senti_decoder(senti_labels,
                                                     encoder_outputs[0],
                                                     attention_mask,
                                                     decoder_input_ids)
        elif task_type == 'ANP_generate':
            labels, decoder_input_ids, decoder_attention_mask = [
                ANP_generate_infos['anp_generate_labels'],
                ANP_generate_infos['anp_generate_decoder_input_ids'],
                ANP_generate_infos['anp_generate_decoder_attention_mask']
            ]
            loss = self.anp_generate_decoder(labels, input_ids,
                                             encoder_outputs[0],
                                             attention_mask, decoder_input_ids,
                                             decoder_attention_mask)
        elif task_type == 'AE_OE':
            spans, span_mask = [
                ae_oe_infos['labels'].to(input_ids.device),
                ae_oe_infos['masks'].to(input_ids.device)
            ]
            state = self.prepare_state(input_ids, image_features,
                                       attention_mask)
            logits = self.span_decoder(spans, state)
            loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])
        else:
            raise RuntimeError("task type error!!!!!!!")

        if task_type == 'Sentiment':
            return loss, predict_senti
        return loss


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first,
                 src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs,
                                                     indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(
                                layer[key1][key2], indices)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new