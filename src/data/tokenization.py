import torch
import numpy as np
from transformers import BartTokenizer
# from src.utils import TaskType


class ConditionTokenizer:
    """
    tokenizer for image features, event and task type
    this is NOT inherent from transformers Tokenizer
    """
    def __init__(self,
                 pretrained_model_name='facebook/bart-base',
                 mlm_token="<<mlm>>",
                 begin_text="<<text>>",
                 end_text="<</text>>",
                 img_feat='<<img_feat>>',
                 begin_img="<<img>>",
                 end_img="<</img>>",
                 ae_token='<<AE>>',
                 oe_token='<<OE>>',
                 aesc_token='<<AESC>>',
                 pos_token='<<POS>>',
                 neu_token='<<NEU>>',
                 neg_token='<<NEG>>',
                 senti_token='<<senti>>',
                 ANP_token='<<ANP>>'):
        self._base_tokenizer = BartTokenizer.from_pretrained(
            pretrained_model_name, )

        self.additional_special_tokens = [
            mlm_token, begin_text, end_text, img_feat, begin_img, end_img,
            senti_token, ANP_token, ae_token, oe_token, aesc_token, pos_token,
            neu_token, neg_token
        ]
        unique_no_split_tokens = self._base_tokenizer.unique_no_split_tokens
        self._base_tokenizer.unique_no_split_tokens = unique_no_split_tokens + self.additional_special_tokens
        self.unique_no_split_tokens = self._base_tokenizer.unique_no_split_tokens
        self._base_tokenizer.add_special_tokens(
            {'additional_special_tokens': self.additional_special_tokens})

        self.mlm_token = mlm_token
        self.begin_text = begin_text
        self.end_text = end_text
        self.img_feat = img_feat
        self.begin_img = begin_img
        self.end_img = end_img
        self.ae_token = ae_token
        self.oe_token = oe_token
        self.senti_token = senti_token
        self.ANP_token = ANP_token

        self.aesc_token = '<<AESC>>'
        self.pos_token = '<<POS>>'
        self.neu_token = '<<NEU>>'
        self.neg_token = '<<NEG>>'

        # self.begin_img_id = self.convert_tokens_to_ids(begin_img)
        # self.end_img_id = self.convert_tokens_to_ids(end_img)

        self.mlm_token_id = self.convert_tokens_to_ids(mlm_token)
        self.begin_text_id = self.convert_tokens_to_ids(begin_text)
        self.end_text_id = self.convert_tokens_to_ids(end_text)
        self.img_feat_id = self.convert_tokens_to_ids(img_feat)
        self.begin_img_id = self.convert_tokens_to_ids(begin_img)
        self.end_img_id = self.convert_tokens_to_ids(end_img)
        self.ae_token_id = self.convert_tokens_to_ids(ae_token)
        self.oe_token_id = self.convert_tokens_to_ids(oe_token)
        self.senti_token_id = self.convert_tokens_to_ids(senti_token)
        self.ANP_token_id = self.convert_tokens_to_ids(ANP_token)

        self.aesc_token_id = self.convert_tokens_to_ids(aesc_token)
        self.pos_token_id = self.convert_tokens_to_ids(pos_token)
        self.neu_token_id = self.convert_tokens_to_ids(neu_token)
        self.neg_token_id = self.convert_tokens_to_ids(neg_token)

        self.vocab_size = self._base_tokenizer.vocab_size
        self.bos_token = self._base_tokenizer.bos_token
        self.bos_token_id = self._base_tokenizer.bos_token_id
        self.eos_token = self._base_tokenizer.eos_token
        self.eos_token_id = self._base_tokenizer.eos_token_id
        self.pad_token = self._base_tokenizer.pad_token
        self.pad_token_id = self._base_tokenizer.pad_token_id
        self.unk_token = self._base_tokenizer.unk_token
        self.unk_token_id = self._base_tokenizer.unk_token_id

        # cur_num_tokens = self._base_tokenizer.vocab_size
        # self.cur_num_tokens = cur_num_tokens
        print('self.bos_token_id', self.bos_token_id)
        print('self.eos_token_id', self.eos_token_id)
        print('self.pad_token_id', self.pad_token_id)
        self.mapping = {
            'AE': '<<AE>>',
            'OE': '<<OE>>',
            'AESC': '<<AESC>>',
            'POS': '<<POS>>',
            'NEU': '<<NEU>>',
            'NEG': '<<NEG>>'
        }
        self.mapping2id = {}
        self.mapping2targetid = {}
        for key, value in self.mapping.items():
            key_id = self._base_tokenizer.convert_tokens_to_ids(
                self._base_tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            # assert key_id[0] >= self.cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid) + 2

    def encode(self, *args, **kwargs):
        return self._base_tokenizer(*args, **kwargs)

    def encode_condition(self, img_num=None, sentence=None):
        """
        tokenize text, image features and event
        the output format (after decoded back):
        task_type [<img> <img_feat> ... <img_feat> </img>] [<event> EVENT </event>] [<mlm> MLM </mlm>]

        :param task_type: str or list[str]
        :param img_num: int or list[int], the number of image features
        :param event: str or list[str], event descriptions
        :param mlm: str or list[str], sentence for masked language modeling
        :return: dict {str: Tensor}, {
                "input_ids": ...,
                "attention_mask": ...,
                "event_mask": ...,          only exist if event is given. 1 for the position with event tokens
                "mlm_mask": ...,            only exist if mlm is given. 1 for the position with mlm tokens
                "img_mask":...,             only exist if img_num is given. 1 for the position with img tokens
            }
        """
        text = []
        for x in img_num:
            text.append('')
        # build task types, a list of
        # <intent>, <before> or <after>
        # if not isinstance(task_type, list):
        #     task_type = [task_type]
        if img_num is not None:
            if not isinstance(img_num, list):
                img_num = [img_num]

            for index, value in enumerate(img_num):
                # print('value', value)
                # print(text[index])
                # print(self.begin_img)
                # print(self.img_feat * value)
                text[
                    index] += self.begin_img + self.img_feat * value + self.end_img

        # build mlm
        # <mlm> MLM </mlm>
        if sentence is not None:
            if not isinstance(sentence, list):
                sentence = [sentence]

            for index, value in enumerate(sentence):
                text[index] += self.begin_text + value + self.end_text
        # print(text[0])
        encoded = self.encode(text,
                              add_special_tokens=False,
                              return_tensors='pt',
                              padding=True)
        # print(encoded['input_ids'][0])
        # build mlm mask
        if sentence is not None:
            sentence_mask = torch.zeros(encoded['input_ids'].size(),
                                        dtype=torch.bool)
            for index, value in enumerate(encoded['input_ids']):
                start = (value == self.begin_text_id).nonzero(as_tuple=True)[0]
                end = (value == self.end_text_id).nonzero(as_tuple=True)[0]
                sentence_mask[index, start + 1:end] = True
            encoded['sentence_mask'] = sentence_mask

        # build img mask
        if img_num is not None:
            encoded['img_mask'] = torch.ones((sentence_mask.size()[0], 36),
                                             dtype=torch.bool)

        return encoded

    def encode_label(self, label, img_num=None):
        text = []

        # build text label
        # <s> LABEL </s>
        if not isinstance(label, list):
            label = [label]

        for value in label:
            text.append(self.bos_token + self.mlm_token + value +
                        self.eos_token)

        # build image features
        # <img> <img_feat> ... <img_feat> </img>
        # if img_num is not None:
        #     if not isinstance(img_num, list):
        #         img_num = [img_num]

        #     for index, value in enumerate(img_num):
        #         text[
        #             index] = self.begin_img + self.img_feat * value + self.end_img + text[
        #                 index]

        encoded_label = self.encode(text,
                                    add_special_tokens=False,
                                    return_tensors='pt',
                                    padding=True)

        input_ids = encoded_label['input_ids']
        attention_mask = encoded_label['attention_mask']

        output_shape = input_ids[:, 1:].shape
        labels = torch.empty(output_shape, dtype=torch.long)
        decoder_input_ids = torch.empty(output_shape, dtype=torch.long)
        decoder_attention_mask = torch.empty(output_shape, dtype=torch.long)

        # remove <s> from labels, remove </s> from decoder_input_ids
        # remove the element in attention_mask at the same position as </s> in decoder_input_ids
        for i in range(labels.size(0)):
            labels[i] = input_ids[i][input_ids[i] != self.bos_token_id]
            decoder_input_ids[i] = input_ids[i][
                input_ids[i] != self.eos_token_id]
            decoder_attention_mask[i] = attention_mask[i][
                input_ids[i] != self.eos_token_id]
        labels[(labels == self.pad_token_id) | (labels == self.begin_img_id) |
               (labels == self.end_img_id) |
               (labels == self.img_feat_id)] = -100
        output = {
            'mlm_labels': labels,
            'mlm_decoder_input_ids': decoder_input_ids,
            'mlm_decoder_attention_mask': decoder_attention_mask
        }

        # build img mask
        # if img_num is not None:
        #     output['label_img_mask'] = labels == self.img_feat_id
        #     output[
        #         'decoder_input_img_mask'] = decoder_input_ids == self.img_feat_id

        return output

    def encode_senti(self, sentis):
        senti_input_text = [
            self.bos_token + self.senti_token for i in range(len(sentis))
        ]
        senti_input_text = self.encode(senti_input_text,
                                       add_special_tokens=False,
                                       return_tensors='pt',
                                       padding=True)
        senti_decoder_input_ids = senti_input_text['input_ids']
        senti_decoder_attention_mask = senti_input_text['attention_mask']

        sentiment = []
        for senti in sentis:
            # senti = x['sentiment']
            sentiment.append(senti)
            # else:
            #     raise ValueError('sentiment label error!!')
        output = {
            'senti_labels': torch.from_numpy(np.array(sentiment)),
            'senti_decoder_input_ids': senti_decoder_input_ids,
            'senti_decoder_attention_mask': senti_decoder_attention_mask
        }
        return output

    def encode_anp_dis(self, batch_size):
        ANP_input_text = [
            self.bos_token + self.ANP_token for i in range(batch_size)
        ]
        ANP_input_text = self.encode(ANP_input_text,
                                     add_special_tokens=False,
                                     return_tensors='pt',
                                     padding=True)
        output = {}
        output['ANP_decoder_input_ids'] = ANP_input_text['input_ids']
        output['ANP_decoder_attention_mask'] = ANP_input_text['attention_mask']

        return output

    def encode_ae(self, aspect_spans, ae_max_len):
        target_shift = len(self.mapping2targetid) + 2
        ae_text = []
        masks = []
        for x in aspect_spans:
            cur_text = [
                0, self.mapping2targetid['AE'], self.mapping2targetid['AE']
            ]
            mask = [0, 0, 1]
            for span in x:
                cur_text.append(span[0] + target_shift)
                cur_text.append(span[1] + target_shift)
                mask.append(1)
                mask.append(1)
            cur_text.append(1)
            mask.append(1)
            cur_text = cur_text + [
                1 for i in range(ae_max_len - len(cur_text))
            ]
            mask = mask + [0 for i in range(ae_max_len - len(mask))]
            # print(cur_text)
            ae_text.append(cur_text)
            masks.append(mask)
        # print(ae_text[0])
        # for xx in ae_text:
        #     if xx == None:
        #         print('aspect shit!!!!!!!!!!!!!!!')
        output = {}
        output['AE_labels'] = torch.tensor(ae_text)
        output['AE_masks'] = torch.tensor(masks)
        # output['AE_masks'][:, 2] = 1

        return output

    def encode_oe(self, opinion_spans, oe_max_len):
        target_shift = len(self.mapping2targetid) + 2
        oe_text = []
        masks = []
        # print(len(opinion_spans))
        for x in opinion_spans:
            cur_text = [
                0, self.mapping2targetid['OE'], self.mapping2targetid['OE']
            ]
            mask = [0, 0, 1]
            for span in x:
                cur_text.append(span[0] + target_shift)
                cur_text.append(span[1] + target_shift)
                mask.append(1)
                mask.append(1)
            cur_text.append(1)
            mask.append(1)
            cur_text = cur_text + [
                1 for i in range(oe_max_len - len(cur_text))
            ]
            mask = mask + [0 for i in range(oe_max_len - len(mask))]
            # print(cur_text)
            oe_text.append(cur_text)
            masks.append(mask)
        output = {}
        # print(oe_text[0], len(oe_text))
        # for xx in oe_text:
        #     if xx == None:
        #         print('opinion shit!!!!!!!!!!!!!!!')
        output['OE_labels'] = torch.tensor(oe_text)
        output['OE_masks'] = torch.tensor(masks)
        return output

    def decode(self, token_ids, skip_special_tokens=False):
        return self._base_tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, tokens):
        return self._base_tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self._base_tokenizer.convert_ids_to_tokens(ids)

    def get_base_tokenizer(self):
        return self._base_tokenizer

    def __len__(self):
        return len(self._base_tokenizer)
