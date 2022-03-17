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
#from src.model.mixins import GenerationMixin, FromPretrainedMixin
from src.model.modules import MultiModalBartEncoder, MultiModalBartDecoder_span, Span_loss


class MultiModalBartModel_AESC(PretrainedBartModel):
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
            print('num_tokens', num_tokens)

            model.resize_token_embeddings(
                len(tokenizer.unique_no_split_tokens) + num_tokens)
            encoder = model.encoder
            decoder = model.decoder

            padding_idx = config.pad_token_id
            encoder.embed_tokens.padding_idx = padding_idx

            # if use_recur_pos:
            #     decoder.set_position_embedding(label_ids[0], tag_first)

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

    def __init__(self, config: MultiModalBartConfig, args, bart_model,
                 tokenizer, label_ids):
        super().__init__(config)
        self.config = config
        label_ids = sorted(label_ids)
        multimodal_encoder, share_decoder = self.build_model(
            args, bart_model, tokenizer, label_ids, config)
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        self.causal_mask = causal_mask.triu(diagonal=1)
        self.encoder = multimodal_encoder
        only_sc = False
        need_tag = True  #if predict the sentiment or not

        self.decoder = MultiModalBartDecoder_span(self.config,
                                                  tokenizer,
                                                  share_decoder,
                                                  tokenizer.pad_token_id,
                                                  label_ids,
                                                  self.causal_mask,
                                                  need_tag=need_tag,
                                                  only_sc=False)
        self.span_loss_fct = Span_loss()

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
        state = BartState(
            encoder_outputs,
            encoder_mask,
            input_ids[:,
                      38:],  #the text features start from index 38, the front are image features.
            first,
            src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(
            self,
            input_ids,
            image_features,
            attention_mask=None,
            aesc_infos=None,
            encoder_outputs: Optional[Tuple] = None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):

        state = self.prepare_state(input_ids, image_features, attention_mask)
        spans, span_mask = [
            aesc_infos['labels'].to(input_ids.device),
            aesc_infos['masks'].to(input_ids.device)
        ]

        logits = self.decoder(spans, state)

        loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])

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
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new
