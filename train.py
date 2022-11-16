from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core import eval, map_eval
from core.utils import AverageMeter, create_logger
import models.loss as loss
import math

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
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
    parser.add_argument('--tensorboard', default=False, action="store_true", help='use tensorboard')
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
    config.tensorboard = args.tensorboard


def get_all_proposals(durations):
    all_proposals = list()
    for duration in durations:
        T = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        proposals = list()
        for i in range(T):
            for j in range(1, T+1):
                if i < j:
                    proposals.append([i*duration/T, j*duration/T])
        all_proposals.append(proposals)

    return all_proposals


def get_all_augmented_proposals(durations):
    all_proposals = list()
    for duration in durations:
        T = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        proposals = list()
        for i in range(T):
            for j in range(1, T+1):
                if i < j:
                    proposals.append([i*duration/T, j*duration/T])
                else:
                    # augmented for all T*T
                    proposals.append([0.0, 0.0])
        all_proposals.append(proposals)

    return all_proposals


def get_iou_local_mask(T, restrict_size):

    def calculate_iou(proposal_1, proposal_2):
        x_max = np.maximum(proposal_1[0], proposal_2[0])
        x_min = np.minimum(proposal_1[0], proposal_2[0])
        y_min = np.minimum(proposal_1[1], proposal_2[1])
        y_max = np.maximum(proposal_1[1], proposal_2[1])
        inter = np.maximum(0.0, y_min - x_max)
        union = y_max - x_min
        iou = inter / (union + 1e-5)
        return iou
    
    RS = (restrict_size - 1) // 2

    T_sq = T * T
    mask = torch.zeros((T_sq, T, T)).cuda()
    
    all_proposals = get_all_augmented_proposals([100,])[0]
    assert len(all_proposals) == T_sq
    
    for ij in range(T_sq):
        i = ij // T
        j = ij % T
        if i > j:
            continue
        else:
            for _i in range(i - RS, i + RS+1):
                for _j in range(j - RS, j + RS+1):
                    if (_i >= 0) and (_i < T) and (_j >= 0) and (_j < T):
                        if _i > _j:
                            continue
                        else:
                            proposal_ij = all_proposals[i*T+j]
                            proposal_ij_N = all_proposals[_i*T+_j]
                            iou_np = calculate_iou(proposal_ij, proposal_ij_N)
                            mask[ij, _i, _j] = torch.tensor(iou_np).cuda()

    mask = mask.reshape(T_sq, T_sq)
    return mask


def get_structure_mask(T):
    T_sq = T * T
    mask = torch.zeros((T_sq, T, T)).cuda()
    neg_mask = torch.zeros((T_sq, T, T)).cuda()
    upper_tri_template = torch.triu(torch.ones(T, T)).cuda()
    for ij in range(T_sq):
        i = ij // T
        j = ij % T
        if i > j:
            continue
        else:
            for _i in range(i, j+1):
                for _j in range(i, j+1):
                    if _i > _j:
                        continue
                    else:
                        mask[ij, _i, _j] = 1

            neg_mask[ij] = upper_tri_template - mask[ij]

    mask = mask.reshape(T_sq, T_sq)
    neg_mask = neg_mask.reshape(T_sq, T_sq)
    return mask, neg_mask


def get_temporal_mask(T):
    T_sq = T * T
    mask = torch.zeros((T_sq, T, T)).cuda()
    neg_mask = torch.zeros((T_sq, T, T)).cuda()
    upper_tri_template = torch.triu(torch.ones(T, T)).cuda()
    for ij in range(T_sq):
        i = ij // T
        j = ij %  T
        if i > j:
            continue
        else:
            for _i in range(i, T):
                for _j in range(0, T):
                    if _i > _j:
                        continue
                    else:
                        mask[ij, _i, _j] = 1

            neg_mask[ij] = upper_tri_template - mask[ij]

    mask = mask.reshape(T_sq, T_sq)
    neg_mask = neg_mask.reshape(T_sq, T_sq)
    return mask, neg_mask


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))

    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writter = SummaryWriter()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME
    
    train_dataset = getattr(datasets, dataset_name)('train', config)
    # if config.TEST.EVAL_TRAIN:
    #     eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val', config)
    test_dataset = getattr(datasets, dataset_name)('test', config)

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint, strict=True)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR, patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)

    # if 'structure_loss' in config.LOSS.NAME:
    structure_mask, neg_structure_mask = get_structure_mask(config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE)
    temporal_mask, neg_temporal_mask = get_temporal_mask(config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE)
    if 'MIL_v5_Hard_SS_v6' in config.LOSS.NAME:
        iou_local_mask = get_iou_local_mask = get_iou_local_mask(config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE, config.LOSS.PARAMS.RESTRICT_SIZE)

    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn(dataset_name, 'train'))
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn(dataset_name, 'test'))
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn(dataset_name, 'test'))
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn(dataset_name, 'test'))
        else:
            raise NotImplementedError

        return dataloader

    def network(sample, warming=False):

        lambda_MIL = config.LOSS.LAMBDA.MIL_LOSS
        lambda_SS_CONST = config.LOSS.LAMBDA.SS_CONST_LOSS
        lambda_CS_CONST = config.LOSS.LAMBDA.CS_CONST_LOSS
        lambda_DualMIL_SELF_DIS = config.LOSS.LAMBDA.DualMIL_SELF_DIS_LOSS

        if not model.training:
            import pdb; pdb.set_trace()
            print('need change the get_proposal_results parts...')

        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        neg_textual_input = sample['batch_neg_word_vectors'].cuda()
        neg_textual_mask = sample['batch_neg_txt_mask'].cuda()
        neg_visual_input = sample['batch_neg_vis_input'].cuda()
        # map_gt = sample['batch_map_gt'].cuda()
        duration = [100,]  # dummy duration for training
        headline_idx = sample['batch_headline_idx']
        article_idx = sample['batch_article_idx']
        neg_headline_idx = sample['batch_neg_headline_idx']
        neg_article_idx = sample['batch_neg_article_idx']
        pos_sent_idx = sample['batch_pos_sent_idx']
        neg_sent_idx = sample['batch_neg_sent_idx']

        merged_prediction, map_mask, pred_ks = model(textual_input, textual_mask, visual_input)
        neg_v_prediction, neg_v_map_mask, neg_v_ks = model(textual_input, textual_mask, neg_visual_input)
        neg_t_prediction, neg_t_map_mask, neg_t_ks = model(neg_textual_input, neg_textual_mask, visual_input)

        loss_value = 0
        loss_value_dict = dict()


        if 'MIL_v5' in config.LOSS.NAME:

            loss_MIL, topk_pos_sent_mask, topk_pos_sent_id = \
                                                    loss.MIL_v5(pos_scores_list=pred_ks,
                                                               pos_masks=map_mask,
                                                               neg_v_scores_list=neg_v_ks,
                                                               neg_v_masks=neg_v_map_mask,
                                                               neg_t_scores_list=neg_t_ks,
                                                               neg_t_masks=neg_t_map_mask,
                                                               pos_textual_masks=textual_mask,
                                                               neg_textual_masks=neg_textual_mask,
                                                               cfg=config.LOSS.PARAMS)

            loss_value += lambda_MIL * loss_MIL
            loss_value_dict['MIL_loss'] = loss_MIL.item()


        if 'Baseline_WSTAN_MIL' in config.LOSS.NAME:

            loss_MIL = loss.Baseline_WSTAN_MIL(pos_scores=merged_prediction,
                                               pos_masks=map_mask,
                                               neg_v_scores=neg_v_prediction,
                                               neg_v_masks=neg_v_map_mask,
                                               neg_t_scores=neg_t_prediction,
                                               neg_t_masks=neg_t_map_mask,
                                               cfg=config.LOSS.PARAMS)

            loss_value += lambda_MIL * loss_MIL
            loss_value_dict['MIL_loss'] = loss_MIL.item()


        if 'Baseline_WSTAN_Self_Dis' in config.LOSS.NAME:

            loss_WSTAN_Self_Dis = loss.Baseline_WSTAN_Self_Dis(scores=pred_ks[0],
                                                               masks=map_mask,
                                                               cfg=config.LOSS.PARAMS)

            loss_value += lambda_WSTAN_SELF_DIS * loss_WSTAN_Self_Dis
            loss_value_dict['Self_Dis_loss'] = loss_WSTAN_Self_Dis.item()


        if 'MIL_v5_Hard_SS' in config.LOSS.NAME:

            loss_MIL_restrict, topk_pos_sent_mask, topk_pos_sent_id  = \
                                        loss.MIL_v5_Hard_SS(pos_scores_list=pred_ks,
                                                               pos_masks=map_mask,
                                                               neg_v_scores_list=neg_v_ks,
                                                               neg_v_masks=neg_v_map_mask,
                                                               neg_t_scores_list=neg_t_ks,
                                                               neg_t_masks=neg_t_map_mask,
                                                               pos_textual_masks=textual_mask,
                                                               neg_textual_masks=neg_textual_mask,
                                                               cfg=config.LOSS.PARAMS)

            loss_value += lambda_SS_CONST * loss_MIL_restrict
            loss_value_dict['MIL_SS_CONST_loss'] = loss_MIL_restrict.item()


        if 'MIL_v5_Hard_SS_v6' in config.LOSS.NAME:

            loss_MIL_restrict, topk_pos_sent_mask, topk_pos_sent_id  = \
                                        loss.MIL_v5_Hard_SS_v6(pos_scores_list=pred_ks,
                                                               pos_masks=map_mask,
                                                               neg_v_scores_list=neg_v_ks,
                                                               neg_v_masks=neg_v_map_mask,
                                                               neg_t_scores_list=neg_t_ks,
                                                               neg_t_masks=neg_t_map_mask,
                                                               pos_textual_masks=textual_mask,
                                                               neg_textual_masks=neg_textual_mask,
                                                               cfg=config.LOSS.PARAMS,
                                                               iou_local_mask=iou_local_mask)

            loss_value += lambda_SS_CONST * loss_MIL_restrict
            loss_value_dict['MIL_SS_CONST_loss'] = loss_MIL_restrict.item()


        if 'MIL_v5_Soft_SS' in config.LOSS.NAME:

            loss_MIL_restrict, topk_pos_sent_mask, topk_pos_sent_id  = \
                                        loss.MIL_v5_Soft_SS(pos_scores_list=pred_ks,
                                                               pos_masks=map_mask,
                                                               neg_v_scores_list=neg_v_ks,
                                                               neg_v_masks=neg_v_map_mask,
                                                               neg_t_scores_list=neg_t_ks,
                                                               neg_t_masks=neg_t_map_mask,
                                                               pos_textual_masks=textual_mask,
                                                               neg_textual_masks=neg_textual_mask,
                                                               cfg=config.LOSS.PARAMS)

            loss_value += lambda_SS_CONST * loss_MIL_restrict
            loss_value_dict['MIL_SS_CONST_loss'] = loss_MIL_restrict.item()


        if 'MIL_v5_CS_v1' in config.LOSS.NAME:

            loss_multi_scale = loss.MIL_v5_CS_v1(scores=pred_ks,
                                                   masks=map_mask,
                                                   sent_idx=pos_sent_idx,
                                                   headline_idx=headline_idx,
                                                   article_idx=article_idx,
                                                   cfg=config.LOSS.PARAMS)
            loss_value += lambda_CS_CONST * loss_multi_scale
            loss_value_dict['multi_scale_loss'] = loss_multi_scale.item()


        if 'MIL_v5_CS_v2' in config.LOSS.NAME:

            loss_multi_scale = loss.MIL_v5_CS_v2(scores=pred_ks,
                                                   masks=map_mask,
                                                   sent_idx=pos_sent_idx,
                                                   headline_idx=headline_idx,
                                                   article_idx=article_idx,
                                                   cfg=config.LOSS.PARAMS)
            loss_value += lambda_CS_CONST * loss_multi_scale
            loss_value_dict['multi_scale_loss'] = loss_multi_scale.item()

        sorted_times = None

        return loss_value, sorted_times, loss_value_dict


    def eval_network(sample):
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        # map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']
        headline_idx = sample['batch_headline_idx']
        article_idx = sample['batch_article_idx']

        merged_prediction, map_mask, pred_ks = model(textual_input, textual_mask, visual_input)


        ###### without careful thinking
        loss_value = 0
        prob_ks = pred_ks

        sorted_times_with_scores = get_proposal_results_by_threshold(prob_ks[-1].squeeze(1), duration)
        all_proposals = get_all_proposals(duration)

        return loss_value, sorted_times_with_scores, prob_ks[-1], all_proposals


    def get_proposal_results_by_threshold(scores, durations):
        top_k = config.TEST.TOP_K
        pred_thresh = config.TEST.PRED_THRESH

        batch_out_sorted_times = []
        for score, duration in zip(scores, durations):
        
            out_sorted_times = []
            for each_score in score:
                T = each_score.shape[-1]
                each_score_np = each_score.cpu().detach().numpy().ravel()
                each_score_np = np.round(each_score_np, 4)
                _sorted_indexs = np.argsort(each_score_np)[::-1]
                sorted_indexs = np.dstack((*np.unravel_index(_sorted_indexs, (T, T)), each_score_np[_sorted_indexs])).tolist()
                sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

                sorted_indexs[:,1] = sorted_indexs[:,1] + 1
                sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
                target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
                sorted_indexs[:, :2] = sorted_indexs[:, :2].float() / target_size * duration

                # filter out predictions less than thresholds (0.5) or 5 predictions for each sentence
                sorted_mask = sorted_indexs[:, 2] > pred_thresh
                sorted_indexs = sorted_indexs[sorted_mask]
                if len(sorted_indexs) > top_k: 
                    sorted_indexs = sorted_indexs[:top_k]
                out_sorted_times.append(sorted_indexs.tolist())

            batch_out_sorted_times.append(out_sorted_times)
        return batch_out_sorted_times


    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset)/config.TRAIN.BATCH_SIZE*config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])


    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)


    def debug_state(state):

        model.eval()

        table_message = ''

        test_state = engine.test(eval_network, iterator('test'), 'test')
        table_message += '\n' + '*********** Segment-level Metrics ***********'
        table_message += '\n' + str(test_state['metrics'])

        # table_message += '\n' + '*********** Video-level Metrics ***********'
        # table_message += '\n' + str(test_state['vid_sent_match'])

        message = table_message+'\n'
        logger.info(pprint.pprint(message))

        saved_model_filename = os.path.join(config.MODEL_DIR,'{}/{}/{}/iter{:06d}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}.pkl'.format(
            dataset_name, model_name+'_'+config.DATASET.VIS_INPUT_TYPE, config.TAG,
            state['t'], test_state['metrics']['recall_100'][0.1], test_state['metrics']['recall_100'][0.3],
                        test_state['metrics']['recall_50'][0.1], test_state['metrics']['recall_50'][0.3],
                        test_state['metrics']['mAP'][0.1], test_state['metrics']['mAP'][0.3]))

        rootfolder1 = os.path.dirname(saved_model_filename)
        os.makedirs(rootfolder1, exist_ok=True)

        import pdb; pdb.set_trace()
        print('chenlong')
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), saved_model_filename)
        else:
            torch.save(model.state_dict(), saved_model_filename)


    def on_update(state):# Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(eval_network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],
                                                   'performance on training set')
                table_message += '\n'+ train_table
            if not config.DATASET.NO_VAL:
                val_state = engine.test(eval_network, iterator('val'), 'val')
                state['scheduler'].step(-val_state['loss_meter'].avg)
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on validation set')
                table_message += '\n'+ val_table

            test_state = engine.test(eval_network, iterator('test'), 'test')
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            table_message += '\n' + '*********** Segment-level Metrics ***********'
            table_message += '\n' + str(test_state['metrics'])

            # table_message += '\n' + '*********** Video-level Metrics ***********'
            # table_message += '\n' + str(test_state['vid_sent_match'])

            message = loss_message+table_message+'\n'
            logger.info(pprint.pprint(message))

            saved_model_filename = os.path.join(config.MODEL_DIR,'{}/{}/{}/iter{:06d}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}.pkl'.format(
                dataset_name, model_name+'_'+config.DATASET.VIS_INPUT_TYPE, config.TAG,
                state['t'], test_state['metrics']['recall_100'][0.1], test_state['metrics']['recall_100'][0.3],
                            test_state['metrics']['recall_50'][0.1], test_state['metrics']['recall_50'][0.3],
                            test_state['metrics']['mAP'][0.1], test_state['metrics']['mAP'][0.3]))


            rootfolder1 = os.path.dirname(saved_model_filename)
            os.makedirs(rootfolder1, exist_ok=True)

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)


            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()


    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        state['all_prediction'] = []
        state['all_proposal'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        # state['loss_meter'].update(state['loss'].item(), 1)
        state['loss_meter'].update(state['loss'], 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)
        state['all_prediction'].extend([state['prediction'][i].detach() for i in batch_indexs])
        state['all_proposal'].extend([state['proposal'][i] for i in batch_indexs])


    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        if dataset_name == 'YouwikiHow':
            sorted_segments_list_with_scores = state['sorted_segments_list']
            all_prediction = state['all_prediction']
            all_proposal = state['all_proposal']
            state['vid_sent_match'], gt_pos_sent = map_eval.eval_wikihow_vidsent(all_prediction, annotations, config.LOSS.PARAMS)
            state['metrics'] = map_eval.eval_wikihow(sorted_segments_list_with_scores, annotations, gt_pos_sent, all_proposal,
                                                        iou_thds=[0.1, 0.3, 0.5], consistent_iou_thds=[0.1, 0.3, 0.5], eval_taskid=False, eval_sampleid=True)
        # elif dataset_name == 'DiDeMo':
        #     state['Rank@N,mIoU@M'], state['miou'] = eval.eval_didemo(state['sorted_segments_list'], annotations, verbose=False)
        # else:
        #     state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations, verbose=False)
        else:
            raise ValueError
        if config.VERBOSE:
            state['progress_bar'].close()


    def tensorboard_writter(state):
        writter.add_scalar('Loss/train', state['loss'].item(), state['t'])
        for k, v in state['loss_value_dict'].items():
            writter.add_scalar('Loss/{}'.format(k), v, state['t'])


    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.hooks['debug_state'] = debug_state ###########
    if config.tensorboard:
        engine.hooks['tensorboard_writter'] = tensorboard_writter
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)