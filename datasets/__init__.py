import torch
import torch.nn as nn
from core.config import config


def pad_sequence(sequences, pad_dim_num=1, batch_first=True, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[pad_dim_num:]
    max_len = []
    for i in range(pad_dim_num):
        max_len.append(max([s.size(i) for s in sequences]))
    if batch_first:
        out_dims = [len(sequences)] + max_len + list(trailing_dims)
    else:
        out_dims = max_len + [len(sequences)] + list(trailing_dims)
    
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[:pad_dim_num]
        if batch_first:
            out_tensor[i, :length[0], :length[1], ...] = tensor
        else:
            out_tensor[:length[0], :length[1], i, ...] = tensor

    return out_tensor


def collate_fn(dataset_name, mode):

    def collate_fn_wikihow_train(batch):

        batch_word_vectors = [b['word_vectors'] for b in batch]
        batch_txt_mask = [b['txt_mask'] for b in batch]
        batch_anno_idxs = [b['anno_idx'] for b in batch]
        batch_vis_feats = [b['visual_input'] for b in batch]
        batch_neg_word_vectors = [b['neg_word_vectors'] for b in batch]
        batch_neg_txt_mask = [b['neg_txt_mask'] for b in batch]
        batch_neg_vis_feats = [b['neg_visual_input'] for b in batch]
        batch_headline_idx = [b['headline_idx'] for b in batch]
        batch_article_idx = [b['article_idx'] for b in batch]
        batch_neg_headline_idx = [b['neg_headline_idx'] for b in batch]
        batch_neg_article_idx = [b['neg_article_idx'] for b in batch]
        batch_pos_sent_idx = [b['pos_sent_idx'] for b in batch]
        batch_neg_sent_idx = [b['neg_sent_idx'] for b in batch]
        batch_video_id = [b['video_id'] for b in batch]
        batch_task_id = [b['task_id'] for b in batch]
        batch_data = {
            'batch_anno_idxs': batch_anno_idxs,
            'batch_word_vectors': pad_sequence(batch_word_vectors, pad_dim_num=2, batch_first=True),
            'batch_txt_mask': pad_sequence(batch_txt_mask, pad_dim_num=2, batch_first=True),
            'batch_neg_word_vectors': pad_sequence(batch_neg_word_vectors, pad_dim_num=2, batch_first=True),
            'batch_neg_txt_mask': pad_sequence(batch_neg_txt_mask, pad_dim_num=2, batch_first=True),
            'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
            'batch_neg_vis_input': nn.utils.rnn.pad_sequence(batch_neg_vis_feats, batch_first=True).float(),
            'batch_headline_idx': batch_headline_idx,
            'batch_article_idx': batch_article_idx,
            'batch_neg_headline_idx': batch_neg_headline_idx,
            'batch_neg_article_idx': batch_neg_article_idx,
            'batch_pos_sent_idx': batch_pos_sent_idx,
            'batch_neg_sent_idx': batch_neg_sent_idx,
            'batch_video_id': batch_video_id,
            'batch_task_id': batch_task_id,
        }
        return batch_data


    def collate_fn_wikihow_test(batch):

        batch_anno_idxs = [b['anno_idx'] for b in batch]
        batch_vis_feats = [b['visual_input'] for b in batch]
        batch_duration = [b['duration'] for b in batch]
        batch_crosstask_anno = [b['crosstask_anno'] for b in batch]
        batch_step2headline = [b['step2headline'] for b in batch]
        batch_headline_idx = [b['headline_idx'] for b in batch]
        batch_article_idx = [b['article_idx'] for b in batch]
        batch_video_id = [b['video_id'] for b in batch]
        batch_task_id = [b['task_id'] for b in batch]

        if 'sentences' in batch[0]: # skip sentence features
            batch_sentences = [b['sentences'] for b in batch]

            batch_data = {
                'batch_anno_idxs': batch_anno_idxs,
                'batch_sentences': batch_sentences,
                'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
                'batch_duration': batch_duration,
                'batch_crosstask_anno': batch_crosstask_anno,
                'batch_step2headline': batch_step2headline,
                'batch_headline_idx': batch_headline_idx,
                'batch_article_idx': batch_article_idx,
                'batch_video_id': batch_video_id,
                'batch_task_id': batch_task_id,
            }

        else:
            batch_word_vectors = [b['word_vectors'] for b in batch]
            batch_txt_mask = [b['txt_mask'] for b in batch]

            batch_data = {
                'batch_anno_idxs': batch_anno_idxs,
                'batch_word_vectors': pad_sequence(batch_word_vectors, pad_dim_num=2, batch_first=True),
                'batch_txt_mask': pad_sequence(batch_txt_mask, pad_dim_num=2, batch_first=True),
                'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
                'batch_duration': batch_duration,
                'batch_crosstask_anno': batch_crosstask_anno,
                'batch_step2headline': batch_step2headline,
                'batch_headline_idx': batch_headline_idx,
                'batch_article_idx': batch_article_idx,
                'batch_video_id': batch_video_id,
                'batch_task_id': batch_task_id,
            }
        return batch_data


    if dataset_name in ['YouwikiHow', 'YouwikiHow_MILNCE']:
        if mode == 'train':
            return collate_fn_wikihow_train
        elif mode == 'test':
            return collate_fn_wikihow_test
    else:
        raise ValueError


def average_to_fixed_length(visual_input):
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS

    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input


from datasets.YouwikiHow import YouwikiHow
from datasets.YouwikiHow_MILNCE import YouwikiHow_MILNCE