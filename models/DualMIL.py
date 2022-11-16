import torch
from torch import nn
import torch.nn.functional as F
from core.config import config
import models.dualmil_modules.frame_modules as frame_modules
import models.dualmil_modules.prop_modules as prop_modules
import models.dualmil_modules.map_modules as map_modules
import models.dualmil_modules.fusion_modules as fusion_modules
from models.dualmil_modules.map_modules.position_encoding import build_position_encoding

class DualMIL(nn.Module):
    def __init__(self):
        super(DualMIL, self).__init__()

        self.frame_layer = getattr(frame_modules, config.DualMIL.FRAME_MODULE.NAME)(config.DualMIL.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.DualMIL.PROP_MODULE.NAME)(config.DualMIL.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.DualMIL.FUSION_MODULE.NAME)(config.DualMIL.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.DualMIL.MAP_MODULE.NAME)(config.DualMIL.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.DualMIL.PRED_INPUT_SIZE, 1, 1, 1)

        self.n_ref = config.DualMIL.N_REF
        if self.n_ref > 0:
            self.map_layer_1 = getattr(map_modules, config.DualMIL.MAP_MODULE.NAME)(config.DualMIL.MAP_MODULE.PARAMS)
            self.pred_layer_1 = nn.Conv2d(config.DualMIL.PRED_INPUT_SIZE, 1, 1, 1)
        if self.n_ref > 1:
            self.map_layer_2 = getattr(map_modules, config.DualMIL.MAP_MODULE.NAME)(config.DualMIL.MAP_MODULE.PARAMS)
            self.pred_layer_2 = nn.Conv2d(config.DualMIL.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, visual_input):

        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)
        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        mapped_h = self.map_layer(fused_h, map_mask)
        tmp_shape = mapped_h.shape
        mapped_h = torch.reshape(mapped_h, (-1, tmp_shape[2], tmp_shape[3], tmp_shape[4]))
        prediction = self.pred_layer(mapped_h)
        prediction = torch.reshape(prediction, (tmp_shape[0], tmp_shape[1],
                                                prediction.shape[-3],
                                                prediction.shape[-2],
                                                prediction.shape[-1]))
        prediction = torch.sigmoid(prediction)
        prediction = prediction * map_mask.unsqueeze(1)


        batch_size = textual_input.shape[0]
        sent_len = textual_input.shape[1]
        textual_mask = torch.reshape(textual_mask, (batch_size, sent_len, -1, 1))
        # tmp_mask = (torch.sum(textual_mask, dim=(-2, -1), keepdim=True) > 0)
        # merged_prediction = torch.sum(prediction, dim=1) / torch.sum(torch.unsqueeze(tmp_mask, -1), dim=1, dtype=torch.float)

        ############ debug ############
        tmp_mask = (torch.sum(textual_mask, dim=(-2, -1)) > 0)
        merged_prediction = [torch.max(prediction[i][tmp_mask[i, :]], dim=0, keepdim=True)[0] for i in range(batch_size)]
        merged_prediction = torch.cat(merged_prediction)

        ### bugs ###
        # merged_prediction = torch.max(prediction, dim=1)[0]
        # refinements
        if self.n_ref == 0:
            return merged_prediction, map_mask, [prediction]
        elif self.n_ref >= 1:
            prediction_1 = self.pred_layer_1(mapped_h)
            prediction_1 = torch.reshape(prediction_1, (tmp_shape[0], tmp_shape[1],
                                                        prediction_1.shape[-3],
                                                        prediction_1.shape[-2],
                                                        prediction_1.shape[-1]))
            prediction_1 = prediction_1 * map_mask.unsqueeze(1)
            if self.n_ref == 1:
                return merged_prediction, map_mask, [prediction, prediction_1]
            else:
                prediction_2 = self.pred_layer_2(mapped_h)
                prediction_2 = torch.reshape(prediction_2, (tmp_shape[0], tmp_shape[1],
                                                            prediction_2.shape[-3],
                                                            prediction_2.shape[-2],
                                                            prediction_2.shape[-1]))
                prediction_2 = prediction_2 * map_mask.unsqueeze(1)

                return merged_prediction, map_mask, [prediction, prediction_1, prediction_2]

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, map_mask
