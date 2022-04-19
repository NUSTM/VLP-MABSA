import torch
import numpy as np
import json
import csv
import os
import json
import torch.utils.data as data

from transformers import AutoTokenizer


class MVSA_Dataset(data.Dataset):
    def __init__(self, infos):
        # print(infos)
        infos = json.load(open(infos, 'r'))
        self.text_dir = infos['text_dir']
        self.img_region_dir = infos['img_region_dir']
        self.senti_dir = infos['senti_dir']
        # self.BIO_dir = infos['BIO_dir']
        self.aspect_span_dict = json.load(open(infos['aspect_span_path'], 'r'))
        self.opinion_span_dict = json.load(
            open(infos['opinion_span_path'], 'r'))
        self.ANP_dir = infos['ANP_dir']
        self.ANP_class_dir = infos['ANP_class_dir']
        self.ANP_class = json.load(open(self.ANP_class_dir, 'r'))
        self.ANP_class = {i: anp for i, anp in enumerate(self.ANP_class)}
        self.ANP2idx = {anp: idx for idx, anp in self.ANP_class.items()}
        self.ANP_len = len(self.ANP_class)
        self.id2senti = json.load(open(self.senti_dir, 'r'))
        self.idx2ANP = json.load(open(self.ANP_dir, 'r'))['images']

        self.create_id2idx()

    def __len__(self):
        return len(self.ids)

    def create_id2idx(self):
        ignore = [
            '3151', '3910', '5995'
        ]  #Pictures of these ids can not be opened, so we remove them.
        self.ids = list(sorted(self.id2senti.keys(), key=lambda x: int(x)))
        for x in ignore:
            self.ids.remove(x)
        self.idx2id = {i: id for i, id in enumerate(self.ids)}

    def get_img_region_box(self, id):
        region_feat = np.load(
            os.path.join(self.img_region_dir + '/_att', id + '.npz'))['feat']
        box = np.load(os.path.join(self.img_region_dir + '/_box', id + '.npy'))

        return region_feat, box

    def process_ANP_distribution(self, distribution):
        result = np.empty([1, self.ANP_len], dtype=float)
        for anp, prob in distribution.items():
            result[0, self.ANP2idx[anp]] = prob

        return result

    def get_ANP_word(self, distribution):
        anp_word = list(distribution.items())[0][0].replace('_', ' ')
        # print(anp_word)
        return anp_word

    def get_img_ANP(self, idx):
        distribution = self.idx2ANP[idx]['bi-concepts']
        words = self.get_ANP_word(distribution)
        dis = self.process_ANP_distribution(distribution)

        return dis, words

    def get_sentiment(self, id):
        sentiment = self.id2senti[id]
        return sentiment

    def get_sentence(self, id):
        sentence = open(os.path.join(self.text_dir,
                                     id + '.txt')).read().strip()
        return sentence

    def get_aspect_spans(self, id):
        aspect_spans = self.aspect_span_dict[id]['aspect_spans']
        return aspect_spans

    def get_opinion_spans(self, id):
        opinion_spans = self.opinion_span_dict[id]['opinion_spans']
        return opinion_spans

    def get_cls(self, id):
        cls_prob = np.load(
            os.path.join(self.img_region_dir + '/_cls_again',
                         id + '.npz'))['feat']
        # _cls = np.argmax(cls_prob, axis=-1)
        return cls_prob

    def __getitem__(self, index):
        output = {}
        data_id = self.idx2id[index]
        region_feat, box = self.get_img_region_box(data_id)
        img_feature = region_feat
        output['img_feat'] = img_feature

        sentence = self.get_sentence(data_id)
        output['sentence'] = sentence

        ANP_dis, ANP_words = self.get_img_ANP(index)
        output['ANP_dis'] = ANP_dis
        output['ANP_words'] = ANP_words

        sentiment = self.get_sentiment(data_id)
        output['sentiment'] = sentiment

        aspect_spans = self.get_aspect_spans(data_id)
        output['aspect_spans'] = aspect_spans

        opinion_spans = self.get_opinion_spans(data_id)
        output['opinion_spans'] = opinion_spans
        output['cls'] = self.get_cls(data_id)
        output['image_id'] = data_id
        output['gt'] = None
        return output


class Twitter_Dataset(data.Dataset):
    def __init__(self, infos, split):
        self.infos = json.load(open(infos, 'r'))

        if split == 'train':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/train.json', 'r'))
            self.img_region_dir = self.infos['img_region_dir'] + '/train'
        elif split == 'dev':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/dev.json', 'r'))
            self.img_region_dir = self.infos['img_region_dir'] + '/dev'
        elif split == 'test':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/test.json', 'r'))
            self.img_region_dir = self.infos['img_region_dir'] + '/test'
        else:
            raise RuntimeError("split type is not exist!!!")

    def __len__(self):
        return len(self.data_set)

    def get_img_region_box(self, id):
        region_feat = np.load(
            os.path.join(self.img_region_dir + '/_att',
                         id[:-4] + '.npz'))['feat']
        box = np.load(
            os.path.join(self.img_region_dir + '/_box', id[:-4] + '.npy'))

        return region_feat, box

    def get_aesc_spans(self, dic):
        aesc_spans = []
        for x in dic:
            aesc_spans.append((x['from'], x['to'], x['polarity']))
        return aesc_spans

    def get_gt_aspect_senti(self, dic):
        gt = []
        for x in dic:
            gt.append((' '.join(x['term']), x['polarity']))
        return gt

    def __getitem__(self, index):
        output = {}
        data = self.data_set[index]
        img_id = data['image_id']
        region_feat, box = self.get_img_region_box(img_id)
        # img_feature = np.concatenate([region_feat, box], axis=1)  #check 维度
        img_feature = region_feat
        output['img_feat'] = img_feature

        output['sentence'] = ' '.join(data['words'])

        aesc_spans = self.get_aesc_spans(data['aspects'])
        output['aesc_spans'] = aesc_spans
        output['image_id'] = img_id
        gt = self.get_gt_aspect_senti(data['aspects'])
        output['gt'] = gt
        return output
