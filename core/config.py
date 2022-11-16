from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

config = edict()

config.WORKERS = 16
config.LOG_DIR = ''
config.MODEL_DIR = ''
config.RESULT_DIR = ''
config.DATA_DIR = ''
config.VERBOSE = False
config.TAG = ''

# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True


# DualMIL related params
config.DualMIL = edict()
config.DualMIL.FRAME_MODULE = edict()
config.DualMIL.FRAME_MODULE.NAME = ''
config.DualMIL.FRAME_MODULE.PARAMS = None
config.DualMIL.PROP_MODULE = edict()
config.DualMIL.PROP_MODULE.NAME = ''
config.DualMIL.PROP_MODULE.PARAMS = None
config.DualMIL.FUSION_MODULE = edict()
config.DualMIL.FUSION_MODULE.NAME = ''
config.DualMIL.FUSION_MODULE.PARAMS = None
config.DualMIL.MAP_MODULE = edict()
config.DualMIL.MAP_MODULE.NAME = ''
config.DualMIL.MAP_MODULE.PARAMS = None
config.DualMIL.PRED_INPUT_SIZE = 512
config.DualMIL.N_REF = 0


# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = '' # The checkpoint for the best performance

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.NAME = ''
config.DATASET.MODALITY = ''
config.DATASET.VIS_INPUT_TYPE = ''
config.DATASET.NO_VAL = False
config.DATASET.BIAS = 0
config.DATASET.NUM_SAMPLE_CLIPS = 256
config.DATASET.TARGET_STRIDE = 16
config.DATASET.DOWNSAMPLING_STRIDE = 16
config.DATASET.SPLIT = ''
config.DATASET.NORMALIZE = False
config.DATASET.RANDOM_SAMPLING = False
config.DATASET.UNSUP = False
config.DATASET.SKIP_SENTENCE_FEATURE = False
config.DATASET.MILNCE_THRE = 0.0

# train
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 20
config.TRAIN.MAX_EPOCH = 20
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False
config.TRAIN.NUM_SAMPLE_SENTENCE = 40

config.LOSS = edict()
config.LOSS.NAME = 'bce_loss'
config.LOSS.PARAMS = None
config.LOSS.LAMBDA = edict()
config.LOSS.LAMBDA.MIL_LOSS = 1.0
config.LOSS.LAMBDA.SS_CONST_LOSS = 1.0
config.LOSS.LAMBDA.CS_CONST_LOSS = 1.0
config.LOSS.LAMBDA.DualMIL_SELF_DIS_LOSS = 1.0
# config.LOSS.LAMBDA.H_STRUCTURE_LOSS = 1.0
# config.LOSS.LAMBDA.T_STRUCTURE_LOSS = 1.0
# config.LOSS.LAMBDA.SELF_RESTRICT_LOSS = 1.0
# config.LOSS.LAMBDA.STRUCTURE_LOSS_H2A = 1.0
# config.LOSS.LAMBDA.STRUCTURE_LOSS_H2H = 1.0

# test
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 0.4
config.TEST.INTERVAL = 1
config.TEST.EVAL_TRAIN = False
config.TEST.BATCH_SIZE = 1
config.TEST.TOP_K = 10
config.TEST.ALL_TOP_K = 10
config.TEST.PRED_THRESH = 0.5


def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))