import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import MaxPool2d, AvgPool2d
from collections import defaultdict

def get_iou(proposal_set1, proposal_set2, mask):

    assert torch.is_tensor(proposal_set1)
    assert torch.is_tensor(proposal_set2)

    x_max = torch.maximum(proposal_set1[:, 0][:, None], proposal_set2[:, 0][None, :])
    x_min = torch.minimum(proposal_set1[:, 0][:, None], proposal_set2[:, 0][None, :])
    y_min = torch.minimum(proposal_set1[:, 1][:, None], proposal_set2[:, 1][None, :])
    y_max = torch.maximum(proposal_set1[:, 1][:, None], proposal_set2[:, 1][None, :])
    inter = torch.maximum(torch.tensor([0.0]).cuda(), y_min - x_max)
    union = y_max - x_min

    iou = inter / (union + 1e-5)

    mask_mat = torch.matmul(mask[:, None], mask[None, :])
    iou = iou * mask_mat

    return iou.float()


def cal_topk_seg_score(scores, topk_seg):
    """
    scores: [Batch_size, 1, T, T]
    return: [Batch_size, topk_seg]
    """
    batch_size = scores.shape[0]
    scores = scores.reshape(batch_size, -1)
    score = torch.topk(scores, topk_seg, dim=1)[0]
    return score.reshape(batch_size, -1)


def cal_topk_sent_score(scores, txt_mask, num_topk_sent):

    """
    scores: [Batch_size, num_sent, topk_seg]
    txt_mask: [Batch_size, num_sent]
    """
    scores = scores * txt_mask.unsqueeze(2)
    scores_over_seg = scores.mean(dim=2)
    topk_idx = torch.topk(scores_over_seg, num_topk_sent, dim=1)[1]
    topk_scores = torch.cat([each_score[each_idx][None, :, :] for each_score, each_idx \
                                                            in zip(scores, topk_idx)], dim=0)
    topk_scores_mask = txt_mask[:, :num_topk_sent]

    return topk_scores, topk_scores_mask, topk_idx


def sentence_video_matching_score(scores, txt_mask, topk_seg, num_topk_sent):
    """
    scores: [Batch_size, num_sent, 1, T, T]
    txt_mask: [Batch_size, num_sent]
    """
    batch_size = scores.shape[0]
    num_sent = scores.shape[1]
    
    all_sent_score = list()
    for sent_idx in range(num_sent):
        sent_scores = scores[:, sent_idx, ...]
        sent_score = cal_topk_seg_score(sent_scores, topk_seg)
        sent_score = sent_score.unsqueeze(1)
        all_sent_score.append(sent_score)

    all_sent_score = torch.cat(all_sent_score, dim=1)
    topk_sent_score, topk_sent_mask, topk_sent_id = cal_topk_sent_score(all_sent_score, txt_mask, num_topk_sent)
    topk_sent_score = topk_sent_score.sum(2).reshape(batch_size, -1)

    return topk_sent_score, topk_sent_mask, topk_sent_id


def headline2article_mapping(headline_i, article_i):

    ## build dict headline_i_2_article_i
    headline2article = defaultdict(list)
    article2headline = dict()
    cursor = 0
    for i, h_i in enumerate(headline_i):
        if h_i == headline_i[-1]:
            headline2article[h_i] = article_i[cursor:]
            for a_j in article_i[cursor:]:
                article2headline[a_j] = h_i
        else:
            for a_j in article_i[cursor:]:
                assert a_j > h_i
                if a_j < headline_i[i+1]:
                    headline2article[h_i].append(a_j)
                    article2headline[a_j] = h_i
                    cursor += 1
                else:
                    break

    return headline2article, article2headline


def MIL_v5(pos_scores_list, pos_masks, neg_v_scores_list, neg_v_masks, 
                             neg_t_scores_list, neg_t_masks, pos_textual_masks, neg_textual_masks, cfg):

    topk_seg = cfg.TOPK_SEG
    topk_sent_ratio = cfg.TOPK_SENT_RATIO

    delta = 0.3 * topk_seg
    loss = 0

    pos_txt_mask = (torch.sum(pos_textual_masks, dim=(-2, -1)) > 0)
    neg_txt_mask = (torch.sum(neg_textual_masks, dim=(-2, -1)) > 0)

    for pos_scores, neg_v_scores, neg_t_scores in zip(pos_scores_list, neg_v_scores_list, neg_t_scores_list):

        num_topk_sent = round(pos_scores.shape[1] * topk_sent_ratio)
        topk_pos_sent_score, topk_pos_sent_mask, topk_pos_sent_id = sentence_video_matching_score(pos_scores, pos_txt_mask, topk_seg, num_topk_sent)
        topk_neg_v_sent_score, topk_neg_v_sent_mask, _ = sentence_video_matching_score(neg_v_scores, pos_txt_mask, topk_seg, num_topk_sent)
        topk_neg_t_sent_score, topk_neg_t_sent_mask, _ = sentence_video_matching_score(neg_t_scores, neg_txt_mask, topk_seg, num_topk_sent)

        # v5 - only calculate the intersection between any two scores
        tmp_0 = torch.zeros((topk_pos_sent_score.shape[0], num_topk_sent, num_topk_sent), dtype=torch.float).cuda()
        neg_v_loss = torch.max(tmp_0, delta - topk_pos_sent_score.unsqueeze(2) + topk_neg_v_sent_score.unsqueeze(1))
        neg_t_loss = torch.max(tmp_0, delta - topk_pos_sent_score.unsqueeze(2) + topk_neg_t_sent_score.unsqueeze(1))
        neg_v_inter_mask = topk_pos_sent_mask.unsqueeze(2) * topk_neg_v_sent_mask.unsqueeze(1)
        neg_t_inter_mask = topk_pos_sent_mask.unsqueeze(2) * topk_neg_t_sent_mask.unsqueeze(1)
        neg_v_loss = neg_v_loss * neg_v_inter_mask
        neg_t_loss = neg_t_loss * neg_t_inter_mask

        loss += torch.sum(neg_v_loss.flatten(-2), dim=1).mean() / topk_seg
        loss += torch.sum(neg_t_loss.flatten(-2), dim=1).mean() / topk_seg

    return loss, topk_pos_sent_mask, topk_pos_sent_id


def Baseline_WSTAN_MIL(pos_scores, pos_masks, neg_v_scores, neg_v_masks, neg_t_scores, neg_t_masks, cfg):

    delta = cfg.DELTA

    def cal_score(scores, masks):
        batch_size = scores.shape[0]
        score = (scores * masks).reshape(batch_size, -1).max(dim=1)[0]
        return score
    
    pos_score = cal_score(pos_scores, pos_masks)
    neg_v_score = cal_score(neg_v_scores, neg_v_masks)
    neg_t_score = cal_score(neg_t_scores, neg_t_masks)

    batch_size = pos_score.shape[0]
    # if not cfg.CROSS_ENTROPY:
    #     tmp_0 = torch.tensor([[0] * batch_size], dtype=torch.float).cuda()
    #     loss = torch.sum(torch.max(tmp_0, delta - pos_score + neg_t_score) + torch.max(tmp_0, delta - pos_score + neg_v_score))
    # else:
    loss = 0
    target_prob = torch.tensor([1] * batch_size, dtype=torch.float).cuda()
    loss += torch.sum(F.binary_cross_entropy(pos_score, target_prob, reduction='mean'))
    target_prob = torch.tensor([0] * batch_size, dtype=torch.float).cuda()
    loss += torch.sum(F.binary_cross_entropy(neg_v_score, target_prob, reduction='mean'))
    loss += torch.sum(F.binary_cross_entropy(neg_t_score, target_prob, reduction='mean'))
    return loss


def Baseline_DualMIL_Self_Dis(scores, masks, cfg):

    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    masks = masks.unsqueeze(1)
    joint_prob = scores * masks

    tmp_shape = scores.shape
    weight_1, targets_tmp = torch.max(joint_prob.flatten(-2), dim=-1)
    # weight_1_detached = weight_1.detach()
    # targets_tmp_detached = targets_tmp.detach()
    targets = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
    targets.scatter_(2, targets_tmp, 1)
    targets = torch.reshape(targets, tmp_shape)

    target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    loss_value = torch.sum(loss * weight_1.unsqueeze(-1).unsqueeze(-1)) / torch.sum(masks)
    return loss_value



def soft_self_restrict_filter(scores, cfg):
    """
    assign soft labels to other overlapped proposals based on IOU
    """
    restrict_kernel = cfg.RESTRICT_SIZE
    restrict_thds = cfg.RESTRICT_THDS

    joint_prob = scores
    tmp_shape = joint_prob.shape

    joint_prob_squeeze = joint_prob.squeeze(2) # [batch_size, num_sent, T, T]

    ##### mask1 for local maximum ####
    max_joint_prob = MaxPool2d(kernel_size=restrict_kernel, stride=1, padding=(restrict_kernel-1)//2).cuda()(joint_prob_squeeze)

    ########### change hard mask to soft version
    # mask1_local_max = (max_joint_prob == joint_prob_squeeze)
    # eariler soft v1
    # mask1_local_max = (joint_prob_squeeze / (max_joint_prob + 1e-5))**2

    # v1
    mask1_local_max = joint_prob_squeeze / (max_joint_prob + 1e-5)

    ##### mask2 for minimum scores ####
    max_scores, targets_tmp = torch.max(joint_prob_squeeze.flatten(-2), dim=-1)
    mask2_by_thds = joint_prob_squeeze > (max_scores * restrict_thds)[:, :, None, None]

    mask_joint_prob = joint_prob * mask1_local_max.unsqueeze(2) * mask2_by_thds.unsqueeze(2)

    return mask_joint_prob



def self_restrict_filter(scores, cfg):
    """
    assign soft labels to other overlapped proposals based on IOU
    """
    restrict_kernel = cfg.RESTRICT_SIZE
    restrict_thds = cfg.RESTRICT_THDS

    joint_prob = scores
    tmp_shape = joint_prob.shape

    joint_prob_squeeze = joint_prob.squeeze(2) # [batch_size, num_sent, T, T]

    ##### mask1 for local maximum ####
    max_joint_prob = MaxPool2d(kernel_size=restrict_kernel, stride=1, padding=(restrict_kernel-1)//2).cuda()(joint_prob_squeeze)
    mask1_local_max = (max_joint_prob == joint_prob_squeeze)

    ##### mask2 for minimum scores ####
    max_scores, targets_tmp = torch.max(joint_prob_squeeze.flatten(-2), dim=-1)
    mask2_by_thds = joint_prob_squeeze > (max_scores * restrict_thds)[:, :, None, None]

    mask_joint_prob = joint_prob * mask1_local_max.unsqueeze(2) * mask2_by_thds.unsqueeze(2)

    return mask_joint_prob


def iou_self_restrict_filter(scores, cfg, iou_local_mask):
    """
    assign soft labels to other overlapped proposals based on IOU
    """
    restrict_kernel = cfg.RESTRICT_SIZE
    restrict_thds = cfg.RESTRICT_THDS

    joint_prob = scores
    joint_prob_squeeze = joint_prob.squeeze(2) # [batch_size, num_sent, T, T]
    tmp_shape = joint_prob_squeeze.shape
    T = tmp_shape[-1]

    ##### mask1 for local maximum ####
    max_joint_prob = MaxPool2d(kernel_size=restrict_kernel, stride=1, padding=(restrict_kernel-1)//2).cuda()(joint_prob_squeeze)
    mask1_local_max = (max_joint_prob == joint_prob_squeeze)

    ### some proposal may have multiple local-maximum neighbors
    iou_weights = AvgPool2d(kernel_size=restrict_kernel, stride=1, padding=(restrict_kernel-1)//2).cuda()(mask1_local_max.float()) * restrict_kernel * restrict_kernel

    mask1_local_by_iou = torch.mm(mask1_local_max.reshape(-1, T*T).float(), iou_local_mask)
    mask1_local_by_iou = mask1_local_by_iou.reshape(tmp_shape)
    mask1_local_by_iou = mask1_local_by_iou / (iou_weights + 1e-5)


    ##### mask2 for minimum scores ####
    max_scores, targets_tmp = torch.max(joint_prob_squeeze.flatten(-2), dim=-1)
    mask2_by_thds = joint_prob_squeeze > (max_scores * restrict_thds)[:, :, None, None]

    mask_joint_prob = joint_prob * mask1_local_by_iou.unsqueeze(2) * mask2_by_thds.unsqueeze(2)

    return mask_joint_prob



def MIL_v5_Soft_SS(pos_scores_list, pos_masks, neg_v_scores_list, neg_v_masks, 
                             neg_t_scores_list, neg_t_masks, pos_textual_masks, neg_textual_masks, cfg):

    topk_seg = cfg.TOPK_SEG
    topk_sent_ratio = cfg.TOPK_SENT_RATIO

    delta = 0.3 * topk_seg
    loss = 0

    pos_txt_mask = (torch.sum(pos_textual_masks, dim=(-2, -1)) > 0)
    neg_txt_mask = (torch.sum(neg_textual_masks, dim=(-2, -1)) > 0)

    assert torch.all(pos_masks == neg_v_masks)
    assert torch.all(pos_masks == neg_t_masks)

    for pos_scores, neg_v_scores, neg_t_scores in zip(pos_scores_list, neg_v_scores_list, neg_t_scores_list):

        self_restrict_pos_scores = soft_self_restrict_filter(pos_scores, cfg)
        self_restrict_neg_v_scores = soft_self_restrict_filter(neg_v_scores, cfg)
        self_restrict_neg_t_scores = soft_self_restrict_filter(neg_t_scores, cfg)

        num_topk_sent = round(pos_scores.shape[1] * topk_sent_ratio)
        topk_pos_sent_score, topk_pos_sent_mask, topk_pos_sent_id = sentence_video_matching_score(self_restrict_pos_scores, pos_txt_mask, topk_seg, num_topk_sent)
        topk_neg_v_sent_score, topk_neg_v_sent_mask, _ = sentence_video_matching_score(self_restrict_neg_v_scores, pos_txt_mask, topk_seg, num_topk_sent)
        topk_neg_t_sent_score, topk_neg_t_sent_mask, _ = sentence_video_matching_score(self_restrict_neg_t_scores, neg_txt_mask, topk_seg, num_topk_sent)

        # v5 - only calculate the intersection between any two scores
        tmp_0 = torch.zeros((topk_pos_sent_score.shape[0], num_topk_sent, num_topk_sent), dtype=torch.float).cuda()
        neg_v_loss = torch.max(tmp_0, delta - topk_pos_sent_score.unsqueeze(2) + topk_neg_v_sent_score.unsqueeze(1))
        neg_t_loss = torch.max(tmp_0, delta - topk_pos_sent_score.unsqueeze(2) + topk_neg_t_sent_score.unsqueeze(1))
        neg_v_inter_mask = topk_pos_sent_mask.unsqueeze(2) * topk_neg_v_sent_mask.unsqueeze(1)
        neg_t_inter_mask = topk_pos_sent_mask.unsqueeze(2) * topk_neg_t_sent_mask.unsqueeze(1)
        neg_v_loss = neg_v_loss * neg_v_inter_mask
        neg_t_loss = neg_t_loss * neg_t_inter_mask

        loss += torch.sum(neg_v_loss.flatten(-2), dim=1).mean() / topk_seg
        loss += torch.sum(neg_t_loss.flatten(-2), dim=1).mean() / topk_seg

    return loss, topk_pos_sent_mask, topk_pos_sent_id


def MIL_v5_Hard_SS(pos_scores_list, pos_masks, neg_v_scores_list, neg_v_masks, 
                             neg_t_scores_list, neg_t_masks, pos_textual_masks, neg_textual_masks, cfg):

    topk_seg = cfg.TOPK_SEG
    topk_sent_ratio = cfg.TOPK_SENT_RATIO
    delta_alpha = cfg.DELTA_ALPHA

    # delta = 0.3 * topk_seg
    delta = delta_alpha * topk_seg

    loss = 0

    pos_txt_mask = (torch.sum(pos_textual_masks, dim=(-2, -1)) > 0)
    neg_txt_mask = (torch.sum(neg_textual_masks, dim=(-2, -1)) > 0)

    assert torch.all(pos_masks == neg_v_masks)
    assert torch.all(pos_masks == neg_t_masks)

    for pos_scores, neg_v_scores, neg_t_scores in zip(pos_scores_list, neg_v_scores_list, neg_t_scores_list):

        self_restrict_pos_scores = self_restrict_filter(pos_scores, cfg)
        self_restrict_neg_v_scores = self_restrict_filter(neg_v_scores, cfg)
        self_restrict_neg_t_scores = self_restrict_filter(neg_t_scores, cfg)

        num_topk_sent = round(pos_scores.shape[1] * topk_sent_ratio)
        topk_pos_sent_score, topk_pos_sent_mask, topk_pos_sent_id = sentence_video_matching_score(self_restrict_pos_scores, pos_txt_mask, topk_seg, num_topk_sent)
        topk_neg_v_sent_score, topk_neg_v_sent_mask, _ = sentence_video_matching_score(self_restrict_neg_v_scores, pos_txt_mask, topk_seg, num_topk_sent)
        topk_neg_t_sent_score, topk_neg_t_sent_mask, _ = sentence_video_matching_score(self_restrict_neg_t_scores, neg_txt_mask, topk_seg, num_topk_sent)

        # v5 - only calculate the intersection between any two scores
        tmp_0 = torch.zeros((topk_pos_sent_score.shape[0], num_topk_sent, num_topk_sent), dtype=torch.float).cuda()
        neg_v_loss = torch.max(tmp_0, delta - topk_pos_sent_score.unsqueeze(2) + topk_neg_v_sent_score.unsqueeze(1))
        neg_t_loss = torch.max(tmp_0, delta - topk_pos_sent_score.unsqueeze(2) + topk_neg_t_sent_score.unsqueeze(1))
        neg_v_inter_mask = topk_pos_sent_mask.unsqueeze(2) * topk_neg_v_sent_mask.unsqueeze(1)
        neg_t_inter_mask = topk_pos_sent_mask.unsqueeze(2) * topk_neg_t_sent_mask.unsqueeze(1)
        neg_v_loss = neg_v_loss * neg_v_inter_mask
        neg_t_loss = neg_t_loss * neg_t_inter_mask

        loss += torch.sum(neg_v_loss.flatten(-2), dim=1).mean() / topk_seg
        loss += torch.sum(neg_t_loss.flatten(-2), dim=1).mean() / topk_seg

    return loss, topk_pos_sent_mask, topk_pos_sent_id


def MIL_v5_Hard_SS_v6(pos_scores_list, pos_masks, neg_v_scores_list, neg_v_masks, 
                             neg_t_scores_list, neg_t_masks, pos_textual_masks, neg_textual_masks, 
                             cfg, iou_local_mask):

    # self restrict based on IoU
    topk_seg = cfg.TOPK_SEG
    topk_sent_ratio = cfg.TOPK_SENT_RATIO
    delta_alpha = cfg.DELTA_ALPHA

    # delta = 0.3 * topk_seg
    delta = delta_alpha * topk_seg

    loss = 0

    pos_txt_mask = (torch.sum(pos_textual_masks, dim=(-2, -1)) > 0)
    neg_txt_mask = (torch.sum(neg_textual_masks, dim=(-2, -1)) > 0)

    assert torch.all(pos_masks == neg_v_masks)
    assert torch.all(pos_masks == neg_t_masks)

    for pos_scores, neg_v_scores, neg_t_scores in zip(pos_scores_list, neg_v_scores_list, neg_t_scores_list):

        self_restrict_pos_scores = iou_self_restrict_filter(pos_scores, cfg, iou_local_mask)
        self_restrict_neg_v_scores = iou_self_restrict_filter(neg_v_scores, cfg, iou_local_mask)
        self_restrict_neg_t_scores = iou_self_restrict_filter(neg_t_scores, cfg, iou_local_mask)

        num_topk_sent = round(pos_scores.shape[1] * topk_sent_ratio)
        topk_pos_sent_score, topk_pos_sent_mask, topk_pos_sent_id = sentence_video_matching_score(self_restrict_pos_scores, pos_txt_mask, topk_seg, num_topk_sent)
        topk_neg_v_sent_score, topk_neg_v_sent_mask, _ = sentence_video_matching_score(self_restrict_neg_v_scores, pos_txt_mask, topk_seg, num_topk_sent)
        topk_neg_t_sent_score, topk_neg_t_sent_mask, _ = sentence_video_matching_score(self_restrict_neg_t_scores, neg_txt_mask, topk_seg, num_topk_sent)

        # v5 - only calculate the intersection between any two scores
        tmp_0 = torch.zeros((topk_pos_sent_score.shape[0], num_topk_sent, num_topk_sent), dtype=torch.float).cuda()
        neg_v_loss = torch.max(tmp_0, delta - topk_pos_sent_score.unsqueeze(2) + topk_neg_v_sent_score.unsqueeze(1))
        neg_t_loss = torch.max(tmp_0, delta - topk_pos_sent_score.unsqueeze(2) + topk_neg_t_sent_score.unsqueeze(1))
        neg_v_inter_mask = topk_pos_sent_mask.unsqueeze(2) * topk_neg_v_sent_mask.unsqueeze(1)
        neg_t_inter_mask = topk_pos_sent_mask.unsqueeze(2) * topk_neg_t_sent_mask.unsqueeze(1)
        neg_v_loss = neg_v_loss * neg_v_inter_mask
        neg_t_loss = neg_t_loss * neg_t_inter_mask

        loss += torch.sum(neg_v_loss.flatten(-2), dim=1).mean() / topk_seg
        loss += torch.sum(neg_t_loss.flatten(-2), dim=1).mean() / topk_seg

    return loss, topk_pos_sent_mask, topk_pos_sent_id



def MIL_v5_CS_v1(scores, masks, sent_idx, headline_idx, article_idx, cfg):

    delta = cfg.DELTA

    sample_loss = list()

    for joint_prob in scores:

        for (single_prob, single_mask, sent_i, headline_i, article_i) in \
                        zip(joint_prob, masks, sent_idx, headline_idx, article_idx):

            loss_item = list()

            headline2article, article2headline = headline2article_mapping(headline_i, article_i)

            for sent_id in sent_i:
                if sent_id in article_i:
                    headline_sent_id = article2headline[sent_id]

                    if headline_sent_id in sent_i:

                        headline_prob = single_prob[sent_i.index(headline_sent_id)]
                        article_prob = single_prob[sent_i.index(sent_id)]

                        tmp_0 = torch.zeros_like(headline_prob)
                        each_loss = torch.max(tmp_0, delta - headline_prob + article_prob)
                        each_loss = each_loss * single_mask
                        loss_item.append(each_loss.sum())

            if len(loss_item) > 0:
                sample_loss.append(torch.sum(torch.stack(loss_item)))

    if len(sample_loss) > 0:
        loss = torch.mean(torch.stack(sample_loss))
        return loss
    else:
        return 0



def MIL_v5_CS_v2(scores, masks, sent_idx, headline_idx, article_idx, cfg):

    delta = cfg.DELTA

    sample_loss = list()

    for joint_prob in scores:

        for (single_prob, single_mask, sent_i, headline_i, article_i) in \
                        zip(joint_prob, masks, sent_idx, headline_idx, article_idx):

            loss_item = list()

            headline2article, article2headline = headline2article_mapping(headline_i, article_i)

            for sent_id in sent_i:
                if sent_id in article_i:
                    headline_sent_id = article2headline[sent_id]

                    if headline_sent_id in sent_i:

                        headline_prob = single_prob[sent_i.index(headline_sent_id)]
                        article_prob = single_prob[sent_i.index(sent_id)]

                        tmp_0 = torch.zeros_like(headline_prob)
                        each_loss = torch.max(tmp_0, delta - headline_prob + article_prob)
                        ##### v2 ########
                        each_loss = each_loss * single_mask * article_prob
                        loss_item.append(each_loss.sum())

            if len(loss_item) > 0:
                sample_loss.append(torch.sum(torch.stack(loss_item)))

    if len(sample_loss) > 0:
        loss = torch.mean(torch.stack(sample_loss))
        return loss
    else:
        return 0


def cal_topk_sent_idx_over_topk_seg(scores, txt_mask):
    """
    txt_mask: mask for dummy sentence
    """

    scores = scores * txt_mask.unsqueeze(2)
    all_topk_scores = list()
    num_sent = scores.shape[1]
    for sent_idx in range(num_sent):
        sent_score = scores[:, sent_idx, :]
        topk_sent_score = cal_topk_seg_score(sent_score)
        topk_sent_score = topk_sent_score.unsqueeze(1)
        all_topk_scores.append(topk_sent_score)
    all_topk_scores = torch.cat(all_topk_scores, dim=1)

    scores_over_seg = all_topk_scores.mean(dim=2)
    topk_sent = round(all_topk_scores.shape[1] * topk_sent_ratio)
    topk_idx = torch.topk(scores_over_seg, topk_sent, dim=1)[1]
    return topk_idx



