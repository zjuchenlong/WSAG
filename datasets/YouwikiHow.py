""" Dataset loader for the YouwikiHow dataset """
import os
import csv

# import h5py
import pickle as pkl
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
from collections import defaultdict

from . import average_to_fixed_length
# from core.eval import iou
# from core.config import config

import nltk

class YouwikiHow(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split, config):
        super(YouwikiHow, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.model_name = config.MODEL.NAME
        self.config = config
        self.skip_sent_feat = config.DATASET.SKIP_SENTENCE_FEATURE

        self.text_tokenizer = nltk.RegexpTokenizer(r"\w+") # remove all punctuation marks with NLTK


        if self.split == 'train':
            anno_file = open(os.path.join(self.data_dir, 'annotations', 'wikihow_data.pkl'), 'rb')
        elif self.split == 'test':
            anno_file = open(os.path.join(self.data_dir, 'annotations', 'crosstask_test.pkl'), 'rb')
        self.raw_annos = pkl.load(anno_file)

        self.task_ids = list(self.raw_annos.keys())
        self.id2fullname = dict()

        self.annotations = list()
        self.all_videos = list()
        self.taskid2video = dict()
        self.video2taskid = dict()
        self.taskid2sents = dict()
        self.taskid2headline_idx = dict()
        self.taskid2article_idx = dict()

        for task_id, anno in tqdm(self.raw_annos.items()):

            # remove these task_id in the test dataset CrossTask
            if task_id in [105222, 44789, 87706]:
                continue

            task_fullname = anno['task_full_name']
            sentences, headline_idx, article_idx = self.reorganize_text(anno['task_text'])
            task_videos = sorted(set(anno['task_video'])) # some duplicated video ids

            self.id2fullname[task_id] = task_fullname
            self.taskid2video[task_id] = list()
            self.taskid2sents[task_id] = sentences
            self.taskid2headline_idx[task_id] = headline_idx
            self.taskid2article_idx[task_id] = article_idx

            for vid in task_videos:
                if vid in ['bHHB3u9pZj4', 'ZJNm0DaKLYs']: # no s3d features from ht100m
                    pass
                else:
                    self.all_videos.append(vid)
                    self.taskid2video[task_id].append(vid)
                    self.video2taskid[vid] = task_id

                    if self.split == 'train':
                        # self.annotations.append({'sentence': sentences, 'video_id': vid, 'task_id': task_id})
                        ######## debug ######
                        self.annotations.append({'sentence': sentences, 'video_id': vid, 'task_id': task_id,
                                                 'headline_idx': headline_idx, 'article_idx': article_idx})
                    elif self.split == 'test':
                        self.annotations.append({'sentence': sentences, 'video_id': vid, 'task_id': task_id,
                                                 'headline_idx': headline_idx, 'article_idx': article_idx})

        if self.split == 'test':
            """More things for evaluation"""
            for anno in self.annotations:
                task_id = anno['task_id']
                vid = anno['video_id']
                vid_duration = self.raw_annos[task_id]['video_duration'][vid]
                anno['duration'] = vid_duration

                raw_crosstask_anno = self.raw_annos[task_id]['crosstask_annotations'][vid]
                anno['crosstask_anno'] = self.reform_crosstask_anno(raw_crosstask_anno)
                anno['step2headline'] = self.raw_annos[task_id]['step2headline']


        if not self.config.online_feat:
            print('Loading all video features ...')
            self.videos_features = dict()
            for vid in tqdm(sorted(set(self.all_videos))[::-1]):
                self.videos_features[vid] = self.get_video_features(vid)


        # #### debug #####
        # if self.split == 'test':
        #     self.annotations = self.annotations[:10]

        # if self.split == 'test':
        #     #### debug for MILNCE ####
        #     new_annotations = list()
        #     for anno in self.annotations:
        #         vid_i = anno['video_id']
        #         NO_FEAT, _ = self.get_video_features(vid_i)
        #         if NO_FEAT == None:
        #             pass
        #         else:
        #             new_annotations.append(anno)
        #     self.annotations = new_annotations


    def __getitem__(self, index):

        sentences = self.annotations[index]['sentence']
        video_id = self.annotations[index]['video_id']
        task_id = self.annotations[index]['task_id']

        if self.split == 'train':
            sentences, sent_idx = self.sample_sentences(sentences)

        if not self.skip_sent_feat:
            word_vectors, txt_mask = self.get_text_features(sentences)
        else:
            assert self.split == 'test', 'only suitable for zero-shot testing'

        if self.config.online_feat:
            ### Read each video sequentially
            visual_input, visual_mask = self.get_video_features(video_id)
        else:
            ### Read all videos at begin
            visual_input, visual_mask = self.videos_features[video_id]

        ### Time scaled to fixed size
        visual_input = average_to_fixed_length(visual_input)
 
        if self.split == 'train':
            if self.model_name == 'DualMIL':
                neg_video_id = random.sample(self.all_videos, 1)[0]
                while self.video2taskid[neg_video_id] == task_id:
                    neg_video_id = random.sample(self.all_videos, 1)[0]

                neg_task_id = random.sample(self.task_ids, 1)[0]
                while neg_task_id == task_id:
                    neg_task_id = random.sample(self.task_ids, 1)[0]
                neg_sentences = self.taskid2sents[neg_task_id]
                neg_sentences, neg_sent_idx = self.sample_sentences(neg_sentences, sample_number=len(sentences))
                neg_word_vectors, neg_txt_mask = self.get_text_features(neg_sentences)

                # neg_sentence is less
                # if len(neg_sentences) < len(sentences):
                #     len_neg_sentences = len(neg_sentences)
                #     sentences = sentences[:len_neg_sentences]
                #     word_vectors = word_vectors[:len_neg_sentences]
                #     txt_mask = txt_mask[:len_neg_sentences]
                #     sent_idx = sent_idx[:len_neg_sentences]


                neg_visual_input, neg_visual_mask = self.get_video_features(neg_video_id)
                neg_visual_input = average_to_fixed_length(neg_visual_input)

                item = {
                    'video_id': video_id,
                    'task_id': task_id,
                    'visual_input': visual_input,
                    'neg_visual_input': neg_visual_input,
                    'anno_idx': index,
                    'word_vectors': word_vectors,
                    'txt_mask': txt_mask,
                    'neg_word_vectors': neg_word_vectors,
                    'neg_txt_mask': neg_txt_mask,
                    'headline_idx': self.annotations[index]['headline_idx'],
                    'article_idx': self.annotations[index]['article_idx'],
                    'neg_headline_idx': self.taskid2headline_idx[neg_task_id],
                    'neg_article_idx': self.taskid2article_idx[neg_task_id],
                    'pos_sent_idx': sent_idx,
                    'neg_sent_idx': neg_sent_idx
                }

                return item

            else:
                raise ValueError

        elif self.split == 'test':

            duration = self.annotations[index]['duration']
            crosstask_anno = self.annotations[index]['crosstask_anno']
            step2headline = self.annotations[index]['step2headline']

            if self.skip_sent_feat:
                item = {
                    'video_id': video_id,
                    'task_id': task_id,
                    'visual_input': visual_input,
                    'anno_idx': index,
                    'sentences': sentences,
                    'duration': duration,
                    'crosstask_anno': crosstask_anno,
                    'step2headline': step2headline,
                    'headline_idx': self.annotations[index]['headline_idx'],
                    'article_idx': self.annotations[index]['article_idx']
                }
            else:
                item = {
                    'video_id': video_id,
                    'task_id': task_id,
                    'visual_input': visual_input,
                    'anno_idx': index,
                    'word_vectors': word_vectors,
                    'txt_mask': txt_mask,
                    'duration': duration,
                    'crosstask_anno': crosstask_anno,
                    'step2headline': step2headline,
                    'headline_idx': self.annotations[index]['headline_idx'],
                    'article_idx': self.annotations[index]['article_idx']
                }
            return item


    def __len__(self):
        return len(self.annotations)


    def sample_sentences(self, sentences_list, sample_number=None):

        # random sample 10 sentences to reduce GPU memory
        if sample_number == None:
            sample_number = self.config.TRAIN.NUM_SAMPLE_SENTENCE
        if len(sentences_list) >= sample_number:
            sampled_idx = np.random.choice(len(sentences_list), sample_number, replace=False)
            # sentence sequence are disorder
            new_sentences_list = [sentences_list[i] for i in list(sampled_idx)]
            return new_sentences_list, list(sampled_idx)
        else:
            return sentences_list, list(range(len(sentences_list)))


    def reorganize_text(self, anno_task_text):

        headline_list = anno_task_text['simplified_headline_list']
        article_list = anno_task_text['simplified_article_list']
        head2sent = anno_task_text['head2sent']
        sent2head = anno_task_text['sent2head']

        # idx start from 0
        headline_idx = list() # the idx of headline_sentence in the sentences
        article_idx = list() # the idx of article_sentence in the sentences
        sentences = list()
        for head_i in range(len(headline_list)):
            sentences.append(headline_list[head_i])
            headline_idx.append(len(sentences) - 1) # start from 0
            sent_idx = head2sent[head_i]
            if type(sent_idx) is list:
                for sent_j in sent_idx:
                    sentences.append(article_list[sent_j])
                    article_idx.append(len(sentences) - 1)

        assert len(sentences) == len(headline_list) + len(article_list)
        return sentences, headline_idx, article_idx


    def reform_crosstask_anno(self, crosstask_anno):

        new_anno = defaultdict(list)
        for anno in crosstask_anno:
            step_id = anno['step_id']
            step_s = float(anno['step_s'])
            step_e = float(anno['step_e'])
            new_anno[int(step_id) - 1].append([step_s, step_e])

        return new_anno


    def get_text_features(self, sentences):

        word_vectors = list()
        txt_mask = list()
        for sent in sentences:
            
            word_list = self.text_tokenizer.tokenize(sent)

            # cut sent length to 25 (0.9091), 20 (0.8129), 30 (0.9538)
            word_list = word_list[:25]

            # original WSTAN randomly select N-1 sentences
            idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in word_list], dtype=torch.long) 
            vectors = self.word_embedding(idxs)
            word_vectors.append(vectors)
            txt_mask.append(torch.ones(vectors.shape[0], 1))
        word_vectors = nn.utils.rnn.pad_sequence(word_vectors, batch_first=True)
        txt_mask = nn.utils.rnn.pad_sequence(txt_mask, batch_first=True)

        return word_vectors, txt_mask


    def get_video_features(self, vid):

        if self.split == 'train' and self.vis_input_type == 's3d':
            feature_path = os.path.join(self.data_dir, 'features', 'train_s3d_features_from_ht100m')
        elif self.split == 'test' and self.vis_input_type == 's3d':
            feature_path = os.path.join(self.data_dir, 'features', 'test_s3d_features_fr16_sz256_nf16')
        elif self.split == 'test' and self.vis_input_type == 's3d_multimodal':
            feature_path = os.path.join(self.data_dir, 'features', 'test_s3d_features_fr16_sz256_nf16_multimodal')
        else:
            raise ValueError

        vid_mp4_path = os.path.join(feature_path, vid + '.mp4.npy')
        vid_webm_path = os.path.join(feature_path, vid + '.webm.npy')
        if os.path.exists(vid_mp4_path):
            features = torch.from_numpy(np.load(vid_mp4_path)).float()
        elif os.path.exists(vid_webm_path):
            features = torch.from_numpy(np.load(vid_webm_path)).float()
        else:
            raise ValueError

        if self.config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        vis_mask = torch.ones((features.shape[0], 1))

        return features, vis_mask
