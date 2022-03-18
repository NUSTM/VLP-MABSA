import warnings

import numpy as np
import torch
from itertools import chain
# from src.utils import TaskType


class Collator:
    """
    The collator for all types of dataset.
    Remember to add the corresponding collation code after adding a new type of task.
    """
    def __init__(self,
                 tokenizer,
                 is_mlm=False,
                 has_label=True,
                 mlm_enabled=False,
                 mrm_enabled=False,
                 senti_enabled=False,
                 ae_enabled=False,
                 oe_enabled=False,
                 ae_oe_enabled=False,
                 aesc_enabled=False,
                 anp_enabled=False,
                 anp_generate_enabled=False,
                 twitter_ae_enabled=False,
                 twitter_sc_enabled=False,
                 text_only=False,
                 mlm_probability=0.0,
                 mrm_probability=0.0,
                 lm_max_len=30,
                 max_img_num=36,
                 max_span_len=20):
        """
        :param tokenizer: ConditionTokenizer
        :param mlm_enabled: bool, if use mlm for language modeling. False for autoregressive modeling
        :param mrm_enabled: bool, if use mrm
        :param rp_enabled: bool, if use relation prediction (VG)
        :param ap_enabled: bool, if use attribute prediction (VG)
        :param mlm_probability: float, probability to mask the tokens
        :param mrm_probability: float, probability to mask the regions
        """
        self._tokenizer = tokenizer
        self._has_label = has_label
        self._is_mlm = is_mlm
        self._mrm_enabled = mrm_enabled
        self._mlm_enabled = mlm_enabled
        self._senti_enabled = senti_enabled
        self._anp_enabled = anp_enabled
        self._anp_generate_enabled = anp_generate_enabled
        self._ae_enabled = ae_enabled
        self._oe_enabled = oe_enabled
        self._ae_oe_enabled = ae_oe_enabled
        self._aesc_enabled = aesc_enabled
        self._twitter_ae_enabled = twitter_ae_enabled
        self._twitter_sc_enabled = twitter_sc_enabled
        self._lm_max_len = lm_max_len
        self._max_img_num = max_img_num
        self._mlm_probability = mlm_probability
        self._mrm_probability = mrm_probability
        self._max_span_len = max_span_len
        self.text_only = text_only
        if mlm_enabled and not has_label:
            raise ValueError(
                'mlm_enabled can not be true while has_label is false. MLM need labels.'
            )

    def _clip_text(self, text, length):
        tokenized = []
        for i, word in enumerate(text.split()):
            if i == 0:
                bpes = self._tokenizer._base_tokenizer.tokenize(word)
            else:
                bpes = self._tokenizer._base_tokenizer.tokenize(
                    word, add_prefix_space=True)
            bpes = self._tokenizer._base_tokenizer.convert_tokens_to_ids(bpes)
            tokenized.append(bpes)
        _tokenized = list(chain(*tokenized))
        return self._tokenizer.get_base_tokenizer().decode(_tokenized[:length])

    def __call__(self, batch):
        batch = [entry for entry in batch if entry is not None]

        image_features = [
            torch.from_numpy(x['img_feat'][:self._max_img_num])
            if 'img_feat' in x else torch.empty(0) for x in batch
        ]

        img_num = [len(x) for x in image_features]

        target = [x['sentence'] for x in batch]
        sentence = list(target)

        encoded_conditions = self._tokenizer.encode_condition(
            img_num=img_num, sentence=sentence, text_only=self.text_only)

        input_ids = encoded_conditions['input_ids']
        output = {}
        if self._is_mlm:
            input_ids = self._mask_tokens(
                inputs=input_ids,
                input_mask=encoded_conditions['sentence_mask'])
        condition_img_mask = encoded_conditions['img_mask']

        if self._mrm_enabled:
            encode_mrm = self._tokenizer.encode_mrm([x['cls'] for x in batch])
            mrm_labels_all = encode_mrm['mrm_labels']
            probability_matrix = torch.full(input_ids.shape,
                                            self._mrm_probability,
                                            dtype=torch.float)
            masked_regions = torch.bernoulli(probability_matrix).bool()
            input_ids[masked_regions
                      & condition_img_mask] = self._tokenizer.cls_token_id
            decoder_input_ids = encode_mrm['mrm_decoder_input_ids']
            for i in range(input_ids.size(0)):
                for j in range(36):
                    if input_ids[i, j + 1] == self._tokenizer.cls_token_id:
                        decoder_input_ids[i, j +
                                          2] = self._tokenizer.cls_token_id
            mrm_labels = []
            for i in range(len(batch)):
                # create mrm_labels
                masked_indices = masked_regions[i][
                    condition_img_mask[i]].nonzero(as_tuple=False)
                mrm_label = mrm_labels_all[i]
                mrm_labels.append(mrm_label[masked_indices].clone())

                if len(image_features[i]) > 0:
                    image_features[i][masked_indices] = torch.zeros(
                        (len(masked_indices), 1, 2048),
                        dtype=image_features[i].dtype)
            MRM = {}
            MRM['mrm_labels'] = mrm_labels
            MRM['mrm_decoder_input_ids'] = decoder_input_ids
            MRM['mrm_masks'] = decoder_input_ids == self._tokenizer.cls_token_id
            MRM['mrm_decoder_attention_mask'] = encode_mrm[
                'mrm_decoder_attention_mask']
            output['MRM'] = MRM
            output['task'] = 'MRM'
        output['input_ids'] = input_ids
        output['attention_mask'] = encoded_conditions['attention_mask']
        output['image_features'] = image_features
        output['input_ids'] = input_ids
        if self._has_label:
            # encode mrm and mlm labels
            if self._mlm_enabled:
                mlm_output = self._tokenizer.encode_label(label=target,
                                                          img_num=img_num)
                output['MLM'] = mlm_output
                output['task'] = 'MLM'

            if self._senti_enabled:
                output['Sentiment'] = self._tokenizer.encode_senti(
                    [x['sentiment'] for x in batch])
                output['task'] = 'Sentiment'

            if self._anp_generate_enabled:
                output['ANP_generate'] = self._tokenizer.encode_anp_generate(
                    [x['ANP_words'] for x in batch])
                output['task'] = 'ANP_generate'
            if self._aesc_enabled:
                output['AESC'] = self._tokenizer.encode_aesc(
                    target, [x['aesc_spans'] for x in batch],
                    self._max_span_len)
                output['task'] = 'AESC'
            if self._ae_oe_enabled:
                output['AE_OE'] = self._tokenizer.encode_ae_oe(
                    target, [x['aspect_spans'] for x in batch],
                    [x['opinion_spans'] for x in batch])
                output['task'] = 'AE_OE'

        output['image_id'] = [x['image_id'] for x in batch]
        output['gt'] = [x['gt'] for x in batch]
        return output

    def _mask_tokens(self, inputs, input_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        :param inputs: torch.LongTensor, batch data
        :param input_mask: torch.Tensor, mask for the batch, False for the position with 0% probability to be masked
        """

        labels = inputs.clone()
        tokenizer = self._tokenizer.get_base_tokenizer()

        # We sample a few tokens in each sequence for masked-LM training
        probability_matrix = torch.full(labels.shape,
                                        self._mlm_probability,
                                        dtype=torch.float)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val,
                                              already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                     dtype=torch.bool),
                                        value=0.0)
        if tokenizer.pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced & input_mask] = tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(tokenizer.vocab_size,
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random & input_mask] = random_words[indices_random
                                                           & input_mask]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs
