import json
import argparse
import numpy as np
from terminaltables import AsciiTable

from collections import defaultdict

# from map_utils import compute_average_precision_detection
# from config import config, update_config

from core.map_utils import compute_average_precision_detection, compute_consistent_detection
from core.config import config, update_config

from models.loss import cal_topk_seg_score, headline2article_mapping
import pickle as pkl

def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


def hierarchical_suppression(current_best_det, sorted_all_dets, headline_idx, article_idx):

    current_sent_id = int(current_best_det[3])

    headline2article, article2headline = headline2article_mapping(headline_idx, article_idx)

    ######### headline_idx -> article_idx ########
    if current_sent_id in headline_idx:
        current_sent_id_child = headline2article[current_sent_id]
        if len(current_sent_id_child) > 0:
            for child_i_sent_id in current_sent_id_child:
                child_i_sent_inds = np.where(sorted_all_dets[:, 3] == child_i_sent_id)[0]
                if len(child_i_sent_inds) > 0:

                    #### break hierarchical constraint ####
                    child_i_bad_sent_mask = np.logical_or(sorted_all_dets[child_i_sent_inds][:, 0] < current_best_det[0],
                                                          sorted_all_dets[child_i_sent_inds][:, 1] > current_best_det[1])
                    child_i_bad_sent_inds = child_i_sent_inds[child_i_bad_sent_mask]

                    ####### score small than the current best (may equal) #####
                    score_constraint_mask = sorted_all_dets[child_i_bad_sent_inds][:, 2] < current_best_det[2]
                    child_i_bad_sent_inds = child_i_bad_sent_inds[score_constraint_mask]

                    if len(child_i_bad_sent_inds) > 0:

                        best_start, best_end = current_best_det[0], current_best_det[1]
                        all_start, all_end = sorted_all_dets[child_i_bad_sent_inds, 0], sorted_all_dets[child_i_bad_sent_inds, 1]

                        bad_iou = (np.maximum(best_start - all_start, 0) + np.maximum(all_end - best_end, 0)) / (np.maximum(all_end, best_end) - np.minimum(all_start, best_start))

                        #### decrease to 0.9 ####
                        sorted_all_dets[child_i_bad_sent_inds, 2] = sorted_all_dets[child_i_bad_sent_inds, 2] * np.exp(- bad_iou**2 / 50000)


    #########  article_idx -> headline_idx ########
    if current_sent_id in article_idx:
        parent_sent_id = article2headline[current_sent_id]
        parent_sent_inds = np.where(sorted_all_dets[:, 3] == parent_sent_id)[0]
        if len(parent_sent_inds) > 0:

            #### break hierarchical constraint ####
            parent_bad_sent_mask = np.logical_or(sorted_all_dets[parent_sent_inds][:, 0] > current_best_det[0],
                                                 sorted_all_dets[parent_sent_inds][:, 1] < current_best_det[1])
            parent_bad_sent_inds = parent_sent_inds[parent_bad_sent_mask]

            ####### score small than the current best (may equal) #####
            score_constraint_mask = sorted_all_dets[parent_bad_sent_inds][:, 2] < current_best_det[2]
            parent_bad_sent_inds = parent_bad_sent_inds[score_constraint_mask]

            if len(parent_bad_sent_inds) > 0:

                best_start, best_end = current_best_det[0], current_best_det[1]
                all_start, all_end = sorted_all_dets[parent_bad_sent_inds, 0], sorted_all_dets[parent_bad_sent_inds, 1]

                bad_iou = (np.maximum(all_start - best_start, 0) + np.maximum(best_end - all_end, 0)) / (np.maximum(all_end, best_end) - np.minimum(all_start, best_start))

                #### decrease to 0.9 ####
                sorted_all_dets[parent_bad_sent_inds, 2] = sorted_all_dets[parent_bad_sent_inds, 2] * np.exp(- bad_iou**2 / 50000)


    return sorted_all_dets


def temporal_suppression(current_best_det, sorted_all_dets, headline_idx, article_idx):


    ### v3 only all temporal relation, weighted ###
    current_sent_id = int(current_best_det[3])
    max_sent_id = max(headline_idx[-1], article_idx[-1])
    if current_sent_id < max_sent_id:
        for next_sent_id in range(current_sent_id+1, max_sent_id):
            next_sent_inds = np.where(sorted_all_dets[:, 3] == next_sent_id)[0]
            if len(next_sent_inds) > 0:

                #### break temporal constraint ####
                next_bad_sent_mask = sorted_all_dets[next_sent_inds][:, 0] < current_best_det[0]
                next_bad_sent_inds = next_sent_inds[next_bad_sent_mask]

                ####### score small than the current best (may equal) #####
                score_constraint_mask = sorted_all_dets[next_bad_sent_inds][:, 2] < current_best_det[2]
                next_bad_sent_inds = next_bad_sent_inds[score_constraint_mask]

                if len(next_bad_sent_inds) > 0:

                    best_start, best_end = current_best_det[0], current_best_det[1]
                    all_start, all_end = sorted_all_dets[next_bad_sent_inds, 0], sorted_all_dets[next_bad_sent_inds, 1]

                    # v3
                    # bad_segment = (best_start - all_start) / (best_end - best_start + 1e-5)

                    # v3.1
                    bad_iou = (best_start - all_start) / (np.maximum(all_end, best_end) - np.minimum(all_start, best_start))

                    #### decrease to 0.9 ####
                    sorted_all_dets[next_bad_sent_inds, 2] = sorted_all_dets[next_bad_sent_inds, 2] * np.exp(- bad_iou**2 / 50000)


    ### v2 only all temporal relation ###
    # max_sent_id = max(headline_idx[-1], article_idx[-1])
    # if current_sent_id < max_sent_id:
    #     for next_sent_id in range(current_sent_id+1, max_sent_id):
    #         next_sent_inds = np.where(sorted_all_dets[:, 3] == next_sent_id)[0]
    #         if len(next_sent_inds) > 0:

    #             #### break temporal constraint ####
    #             next_bad_sent_mask = sorted_all_dets[next_sent_inds][:, 0] < current_best_det[0]
    #             next_bad_sent_inds = next_sent_inds[next_bad_sent_mask]

    #             if len(next_bad_sent_inds) > 0:
    #                 #### decrease to 0.9 ####
    #                 sorted_all_dets[next_bad_sent_inds, 2] = sorted_all_dets[next_bad_sent_inds, 2] * 0.99999


    ### v1 only headline temporal relation ###
    # if current_sent_id in headline_idx:
    #     current_i = headline_idx.index(current_sent_id)
    #     if current_i < len(headline_idx):
    #         for next_i in range(current_i+1, len(headline_idx)):
    #             next_sent_id = headline_idx[next_i]
    #             next_sent_inds = np.where(sorted_all_dets[:, 3] == next_sent_id)[0]
    #             if len(next_sent_inds) > 0:

    #                 #### break temporal constraint ####
    #                 next_bad_sent_mask = sorted_all_dets[next_sent_inds][:, 0] < current_best_det[0]
    #                 next_bad_sent_inds = next_sent_inds[next_bad_sent_mask]

    #                 #### decrease to 0.9 ####
    #                 sorted_all_dets[next_bad_sent_inds, 2] = sorted_all_dets[next_bad_sent_inds, 2] * 0.99999

    return sorted_all_dets


def structure_nms(dets, thresh=0.4, top_k=-1, all_top_k=-1, headline_idx=None, article_idx=None,
                    temporal_supp=False, hierarchical_supp=False):

    all_dets = list()
    for i, det_i in enumerate(dets):
        if len(det_i) > 0:
            det_i = np.array(det_i)
            det_i = np.concatenate((det_i, np.ones((len(det_i), 1)) * i), axis=1)
            all_dets.append(det_i)
    if len(all_dets) == 0:
        ret_dict = defaultdict(list)
        return ret_dict
    all_dets = np.concatenate(all_dets)


    keep_dets = list()
    sorted_all_dets = all_dets[np.argsort(all_dets[:, 2])[::-1]]

    while len(sorted_all_dets) > 0:

        current_best_det = sorted_all_dets[0]

        keep_dets.append(current_best_det)

        if len(keep_dets) == all_top_k:
            break

        x1 = sorted_all_dets[:, 0]
        x2 = sorted_all_dets[:, 1]
        lengths = x2 - x1

        xx1 = np.maximum(x1[0], x1[1:])
        xx2 = np.minimum(x2[0], x2[1:])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[0] + lengths[1:] - inter)

        iou_inds = np.where(ovr > thresh)[0] + 1
        same_sent_inds = np.where(sorted_all_dets[:, 3] == sorted_all_dets[0, 3])[0]
        nms_inds = np.intersect1d(iou_inds, same_sent_inds)

        keep_inds = np.setdiff1d(np.arange(1, len(sorted_all_dets)), nms_inds)
        sorted_all_dets = sorted_all_dets[keep_inds]

        if temporal_supp:
            sorted_all_dets = temporal_suppression(current_best_det, sorted_all_dets, headline_idx, article_idx)
        if hierarchical_supp:
            sorted_all_dets = hierarchical_suppression(current_best_det, sorted_all_dets, headline_idx, article_idx)

        sorted_all_dets = sorted_all_dets[np.argsort(sorted_all_dets[:, 2])[::-1]]


    keep_dets = np.array(keep_dets)
    ret_dict = defaultdict(list)
    for sent_i, det_i in zip(keep_dets[:, 3], keep_dets[:, :3]):
        ret_dict[int(sent_i)].append(det_i.tolist())

    return ret_dict


def get_max_iou(gt_win, proposal):

    if isinstance(gt_win, list):
        gt_win = np.array(gt_win)
    if isinstance(proposal, list):
        proposal = np.array(proposal)

    x_max = np.maximum(gt_win[0], proposal[:, 0])
    x_min = np.minimum(gt_win[0], proposal[:, 0])
    y_min = np.minimum(gt_win[1], proposal[:, 1])
    y_max = np.maximum(gt_win[1], proposal[:, 1])
    inter = np.maximum(0.0, y_min - x_max)
    union = y_max - x_min

    iou = inter / union
    argmax = iou.argsort()[-1]
    return argmax, iou[argmax]


def compose_triplet(predictions, ground_truth):

    annoid_to_taskid = dict()
    taskid_to_annoid = defaultdict(list)

    pred_id2data = defaultdict(list)
    for d in predictions:

        pred_windows = d['pred_windows']
        anno_id = d['anno_id']
        for k, v in pred_windows.items():
            for w in v:
                pred_id2data[anno_id].append({
                    'video-id': d['anno_id'],  # in order to use the API
                    'sent-id': k,
                    't-start': w[0],
                    't-end': w[1],
                    'score': w[2]
                })

    gt_id2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d['gt_windows']
        anno_id = d['anno_id']
        task_id = d['task_id']

        annoid_to_taskid[anno_id] = task_id
        taskid_to_annoid[task_id].append(anno_id)
        
        for k, v in gt_windows.items():
            for w in v:
                gt_id2data[anno_id].append({
                    'video-id': d['anno_id'],
                    'sent-id': k,
                    't-start': w[0],
                    't-end': w[1],
                })
    data_triples = [[anno_id, gt_id2data[anno_id], pred_id2data[anno_id]] for anno_id in gt_id2data]

    return data_triples, annoid_to_taskid, taskid_to_annoid


def compute_average_precision_detection_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    mAP_scores, recall_k_scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, mAP_scores, recall_k_scores


def eval_mAP(predictions, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10), eval_taskid=False, eval_sampleid=False):

    data_triples, annoid_to_taskid, taskid_to_annoid = compose_triplet(predictions, ground_truth)

    id2ap_list = {}
    id2recall_list = {}
    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    # for data_triple in data_triples:
    for i, data_triple in enumerate(data_triples):
        
        # skip samples without any ground-truth annotations
        if len(data_triple[1]) == 0:
            continue
        anno_id, mAP_scores, recall_k_scores = compute_ap_from_triple(data_triple)

        id2ap_list[anno_id] = mAP_scores
        id2recall_list[anno_id] = recall_k_scores

    ap_array = np.array(list(id2ap_list.values()))  # (#queries, #thd)
    
    # some samples without any ground-truth annotations
    annoid_2_inds = dict()
    for anno_id in id2ap_list.keys():
        annoid_2_inds[anno_id] = len(annoid_2_inds)

    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip(iou_thds, [np.round(100*e, 2) for e in ap_thds]))
    iou_thd2ap["average"] = np.round(np.mean(ap_thds)*100, 2)

    recall_array = np.array(list(id2recall_list.values()))  # (#queries, #thd)
    recall_thds = recall_array.mean(0)
    iou_thd2recall = dict(zip(iou_thds, [np.round(100*e, 2) for e in recall_thds]))

    if eval_taskid:
        taskid_to_ap = dict()
        taskid_to_recall = dict()
        #### calculate ap/recall over different task_id ####
        for task_id in taskid_to_annoid.keys():

            task_inds = list()
            for i in taskid_to_annoid[task_id]:
                if i in annoid_2_inds:
                    task_inds.append(annoid_2_inds[i])
            
            task_ap_array = ap_array[task_inds]
            task_ap_thds = task_ap_array.mean(0)
            task_iou_thd2ap = dict(zip(iou_thds, [np.round(100*e, 2) for e in task_ap_thds]))
            taskid_to_ap[task_id] = task_iou_thd2ap

            task_recall_array = recall_array[task_inds]
            task_recall_thds = task_recall_array.mean(0)
            task_iou_thd2recall = dict(zip(iou_thds, [np.round(100*e, 2) for e in task_recall_thds]))
            taskid_to_recall[task_id] = task_iou_thd2recall

        return iou_thd2ap, iou_thd2recall, taskid_to_ap, taskid_to_recall, None
    else:
        if eval_sampleid:
            # just for visualization
            sampleid_to_recall = dict()
            for i in range(len(recall_array)):
                recall05 = recall_array[i, 2]
                pred_i = predictions[i]
                task_id = ground_truth[i]['task_id']
                sampleid_to_recall[i] = {'recall05': recall05, 'prediction': pred_i, 'task_id': task_id}
            return iou_thd2ap, iou_thd2recall, None, None, sampleid_to_recall
        else:
            return iou_thd2ap, iou_thd2recall, None, None, None


def compute_consistent_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    consistent_scores = compute_consistent_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, consistent_scores


def eval_consistent(predictions, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):

    data_triples, _, _ = compose_triplet(predictions, ground_truth)

    id2consistent_list = {}
    from functools import partial
    compute_consistent_from_triple = partial(
        compute_consistent_wrapper, tiou_thresholds=iou_thds)

    # for data_triple in data_triples:
    for i, data_triple in enumerate(data_triples):
        
        # skip samples without any ground-truth annotations
        if len(data_triple[1]) == 0:
            continue
        anno_id, consistent_scores = compute_consistent_from_triple(data_triple)
        id2consistent_list[anno_id] = consistent_scores

    consistent_array = np.array(list(id2consistent_list.values()))  # (#queries, #thd)
    consistent_thds = consistent_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2consistent = dict(zip(iou_thds, [np.round(100*e, 2) for e in consistent_thds]))

    return iou_thd2consistent



def eval_wikihow(segments, data, gt_pos_sent, all_proposals, iou_thds=np.linspace(0.1, 0.9, 9), consistent_iou_thds=np.linspace(0.1, 0.9, 9), eval_taskid=False, eval_sampleid=False):

    """ 
    High-level (headline) sentences: recall@K, AP
    Low-level (article) sentences: 

    segments: prediction
    data: ground-truth annotations

    """
    # import pdb; pdb.set_trace()
    # print('chenlong')

    eval_results = {}
    all_top_k_list = config.TEST.ALL_TOP_K.copy()
    all_top_k_list.extend(['ALL'])

    for all_top_k in all_top_k_list:

        all_headline_prediction = list()
        all_article_prediction = list()
        gt_prediction = list()
        upper_gt_prediction = list()
        gt = list()
        article_gt = list()

        for anno_id, (seg, dat, proposal) in enumerate(zip(segments, data, all_proposals)):

            headline_idx = dat['headline_idx']
            article_idx = dat['article_idx']

            ## get ground truth for each sentence
            ########### May need change here ############
            
            # propagate crosstask annotations to wikihow headline sentences
            pseudo_headline_anno = defaultdict(list)
            for step_i, mapped_hl_i in enumerate(dat['step2headline']):
                if mapped_hl_i == '_':
                    continue

                mapped_hl_i = mapped_hl_i.split('/')
                assert isinstance(mapped_hl_i, list)
                if isinstance(mapped_hl_i, list):
                    for h_i in mapped_hl_i:
                        if len(dat['crosstask_anno'][step_i]) > 0:
                            pseudo_headline_anno[headline_idx[int(h_i)]].extend(dat['crosstask_anno'][step_i])

            gt_windows = dict(pseudo_headline_anno)

            headline2article, article2headline = headline2article_mapping(headline_idx, article_idx)
            article_gt_windows = dict()
            for hl_i in gt_windows:
                ar_i = headline2article[hl_i]
                if len(ar_i) > 0:
                    for ar_i_j in ar_i:
                        article_gt_windows[ar_i_j] = gt_windows[hl_i]


            ########### headline/article each with topk #############
            # headline_pred_windows = dict()
            # for hl_i in headline_idx:
            #     seg_i = seg[hl_i]
            #     if len(seg_i) > 0:
            #         seg_i = nms(seg_i, thresh=config.TEST.NMS_THRESH, top_k=config.TEST.TOP_K)
            #         seg_i = [list(seg_ij) for seg_ij in seg_i]
            #     headline_pred_windows[hl_i] = seg_i

            # article_pred_windows = dict()
            # for ar_i in article_idx:
            #     seg_i = seg[ar_i]
            #     if len(seg_i) > 0:
            #         seg_i = nms(seg_i, thresh=config.TEST.NMS_THRESH, top_k=config.TEST.TOP_K)
            #         seg_i = [list(seg_ij) for seg_ij in seg_i]
            #     article_pred_windows[ar_i] = seg_i

            # all_headline_pred_windows = filter_top_K(headline_pred_windows, all_top_k)
            # all_article_pred_windows = filter_top_K(article_pred_windows, all_top_k)
            # gt_pred_windows = filter_top_K(headline_pred_windows, all_top_k, list(gt_pos_sent[anno_id]))
            ########################################################

            pred_windows = dict()
            for sent_i in headline_idx + article_idx:
                seg_i = seg[sent_i]
                if len(seg_i) > 0:
                    seg_i = nms(seg_i, thresh=config.TEST.NMS_THRESH, top_k=config.TEST.TOP_K)
                    seg_i = [list(seg_ij) for seg_ij in seg_i]
                pred_windows[sent_i] = seg_i
            # all_pred_windows = filter_top_K(pred_windows, all_top_k)

            all_pred_windows = structure_nms(seg, thresh=config.TEST.NMS_THRESH, top_k=config.TEST.TOP_K,
                                              all_top_k=all_top_k, headline_idx=headline_idx, article_idx=article_idx,
                                              temporal_supp=False, hierarchical_supp=False)

            all_headline_pred_windows = dict()
            all_article_pred_windows = dict()
            for sent_i in all_pred_windows.keys():
                if sent_i in headline_idx:
                    all_headline_pred_windows[sent_i] = all_pred_windows[sent_i]
                elif sent_i in article_idx:
                    all_article_pred_windows[sent_i] = all_pred_windows[sent_i]
            gt_pred_windows = filter_top_K(pred_windows, all_top_k, list(gt_pos_sent[anno_id]))


            # # calculate the upper-bound for current proposal settings
            # upper_gt_windows = dict()
            # for gt_sent, gt_win_list in gt_windows.items():
            #     upper_gt_windows[gt_sent] = list()
            #     for gt_win in gt_win_list:
            #         matched_idx, propsal_score = get_max_iou(gt_win, proposal)
            #         matched_proposal = proposal[matched_idx]
            #         upper_gt_windows[gt_sent].append([matched_proposal[0], matched_proposal[1], propsal_score])


            all_headline_prediction.append({'anno_id': anno_id, 'pred_windows': all_headline_pred_windows})
            gt_prediction.append({'anno_id': anno_id, 'pred_windows': gt_pred_windows})
            all_article_prediction.append({'anno_id': anno_id, 'pred_windows': all_article_pred_windows})
            # upper_gt_prediction.append({'anno_id': anno_id, 'pred_windows': upper_gt_windows})
            gt.append({'anno_id': anno_id, 'gt_windows': gt_windows, 'task_id': dat['task_id']})
            article_gt.append({'anno_id': anno_id, 'gt_windows': article_gt_windows, 'task_id': dat['task_id']})


        all_mAP, all_recall, taskid_to_ap, taskid_to_recall, sampleid_to_recall = eval_mAP(all_headline_prediction, gt, iou_thds=iou_thds, eval_taskid=eval_taskid, eval_sampleid=eval_sampleid)
        gt_mAP, gt_recall, gt_taskid_to_ap, gt_taskid_to_recall, sampleid_to_recall = eval_mAP(gt_prediction, gt, iou_thds=iou_thds, eval_taskid=eval_taskid, eval_sampleid=eval_sampleid)
        # upper_gt_mAP, upper_gt_recall = eval_mAP(upper_gt_prediction, gt, iou_thds=iou_thds)
        consistent_scores = eval_consistent(all_article_prediction, article_gt, iou_thds=consistent_iou_thds)

        if all_top_k == 'ALL':
            eval_results['mAP'] = all_mAP
            eval_results['gt_mAP'] = gt_mAP
        else:
            eval_results[f'recall_{all_top_k}'] = all_recall
            eval_results[f'gt_recall_{all_top_k}'] = gt_recall
            eval_results[f'consistent_{all_top_k}'] = consistent_scores
            if eval_taskid:
                eval_results[f'taskid_to_ap_{all_top_k}'] = taskid_to_ap
                eval_results[f'taskid_to_recall_{all_top_k}'] = taskid_to_recall

        # eval_results[f'upper_gt_recall_{all_top_k}'] = upper_gt_recall
        # eval_results[f'upper_gt_mAP_{all_top_k}'] = upper_gt_mAP
        # with open('top50_iou05_full.pkl', 'wb') as f:
        # with open('top50_iou05_baseline.pkl', 'wb') as f:
        # with open('top50_iou05_SS.pkl', 'wb') as f:
        # with open('top50_iou05_CS.pkl', 'wb') as f:
        #     pkl.dump(sampleid_to_recall, f)
        # import pdb; pdb.set_trace()

    return eval_results



def eval_wikihow_vidsent(predictions, data, cfg=None, score_thds=None):

    if not score_thds:
        score_thds = cfg.SCORE_THDS
    topk_seg = cfg.TOPK_SEG

    recall_list = list()
    precision_list = list()
    F1_list = list()

    gt_pos_sent = dict()
    pred_pos_sent = dict()
    for idx, (pred, dat) in enumerate(zip(predictions, data)):

        headline_idx = dat['headline_idx']

        ## get ground truth for each sentence
        ########### May need change here ############
        
        # propagate crosstask annotations to wikihow headline sentences
        pseudo_headline_anno = defaultdict(list)
        for step_i, mapped_hl_i in enumerate(dat['step2headline']):
            if mapped_hl_i == '_':
                continue

            mapped_hl_i = mapped_hl_i.split('/')
            assert isinstance(mapped_hl_i, list)
            if isinstance(mapped_hl_i, list):
                for h_i in mapped_hl_i:
                    if len(dat['crosstask_anno'][step_i]) > 0:
                        pseudo_headline_anno[headline_idx[int(h_i)]].extend(dat['crosstask_anno'][step_i])

        gt_windows = dict(pseudo_headline_anno)
        

        ####### different from eval_wikihow ###############
        pos_sent = np.array(list(gt_windows.keys()))
        neg_sent = np.array([i for i in range(len(dat['sentence']) - 1) if i not in pos_sent])


        ######### score of the global segment
        # pred_score = pred[:, 0, -1, -1]
        ######### score of the segment with maximum score
        # num_sent = pred.shape[0]
        # pred_score = pred.reshape(num_sent, -1).max(dim=1)[0]
        # pred_idx = np.where(pred_score.cpu().numpy() > score_thds)[0]

        #########
        pred_score = list()
        num_sent = pred.shape[1]
        for sent_idx in range(num_sent):
            sent_topk_seg_score = cal_topk_seg_score(pred[:, sent_idx, ...], topk_seg)
            pred_score.append(sent_topk_seg_score.unsqueeze(1).cpu().numpy())
        pred_score = np.concatenate(pred_score, axis=1)
        pred_score = pred_score.mean(2).mean(0) # dim=0 means multiple predictions
        pred_idx = np.where(pred_score > score_thds)[0]


        true_pos_idx = np.intersect1d(pos_sent, pred_idx)
        if len(pos_sent) == 0:
            sample_recall = 0
        else:
            sample_recall = len(true_pos_idx) / len(pos_sent)

        if len(pred_idx) == 0:
            sample_precision = 0
        else:
            sample_precision = len(true_pos_idx) / len(pred_idx)
        sample_F1 = 2 * sample_recall * sample_precision / (sample_recall + sample_precision + 1e-5)

        recall_list.append(sample_recall)
        precision_list.append(sample_precision)
        F1_list.append(sample_F1)

        gt_pos_sent[idx] = pos_sent
        pred_pos_sent[idx] = pred_idx


    avg_recall = np.mean(recall_list) 
    avg_precision = np.mean(precision_list)
    avg_F1 = np.mean(F1_list)

    return {'recall': avg_recall * 100, 'precision': avg_precision * 100, 'F1': avg_F1 * 100}, gt_pos_sent


def filter_top_K(pred_windows, all_top_k, sent_prior=None):

    if isinstance(sent_prior, list):
        filtered_pred_windows = dict()
        for hl_i, seg_i in pred_windows.items():
            if hl_i not in sent_prior:
                continue
            filtered_pred_windows[hl_i] = seg_i
        pred_windows = filtered_pred_windows

    scores_list = list()
    for hl_i, seg_i in pred_windows.items():
        if len(seg_i) > 0:
            for seg_ij in seg_i:
                scores_list.append(seg_ij[2])

    if all_top_k == 'ALL':
        return pred_windows
    else:
        if len(scores_list) < all_top_k:
            return pred_windows
        else:
            cutoff_score = sorted(scores_list)[-all_top_k]

            filtered_windows = dict()
            filtered_list_size = 0
            for hl_i, seg_i in pred_windows.items():

                if filtered_list_size == all_top_k:
                    break

                if len(seg_i) == 0:
                    filtered_windows[hl_i] = seg_i
                else:
                    filtered_windows[hl_i] = list()
                    for seg_ij in seg_i:
                        if seg_ij[2] >= cutoff_score:
                            filtered_windows[hl_i].append(seg_ij)
                            filtered_list_size += 1
                            if filtered_list_size == all_top_k:
                                break

            return filtered_windows


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--online_feat', default=False, action="store_true", help='load feature online')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag
    config.online_feat = args.online_feat


def eval_main():

    # prediction = [{'anno_id': 100, 
    #               'pred_windows': {0: [[0, 10, 0.98], [20, 30, 0.88]], 2: [[10, 20, 0.6]]}
    #               }]
    # gt = [{'anno_id': 100,
    #       'gt_windows': {0: [[0, 10], [20, 25]], 1: [[10, 20]]}
    #       }]

    # results = eval_mAP(prediction, gt)


    import pickle as pkl
    temp = pkl.load(open('/dvmm-filer3a/users/chenlong/Projects/ws-ms-grounding/temp.pkl', 'rb'))

    segments = temp['segments']
    data = temp['data']

    # results = eval_wikihow(segments, data, iou_thds=np.linspace(0.05, 0.95, 19))
    results = eval_wikihow(segments, data, iou_thds=np.linspace(0.1, 0.9, 9))
    print(results)


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    eval_main()