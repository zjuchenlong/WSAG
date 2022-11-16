# WSAG: Weakly-Supervised Temporal Article Grounding

This repo contains the training and evaluation codes for WSAG. The codes are built based on [2DTAN](https://github.com/researchmm/2D-TAN) and [WSTAN](https://github.com/ycWang9725/WSTAN).


## Training

Based on the model training experience, the model should be trained with multiple stages:
1. Train with only MIL loss:
```
python train.py \
        --cfg ./configs/YouwikiHow/16x16_debug.yaml \
        --tag EXP_TAG --verbose --online_feat
```
2. Finetune with single-sentence constraint. (change the model path in the config file: MODEL/CHECKPOINT)
```
python train.py \
         --cfg ./configs/YouwikiHow/16x16_MIL_Hard_SS_K7D05_continue2.yaml \
         --tag MIL_16x16_Hard_SS_K7D05_melon --verbose --online_feat \
         --tensorboard
```
3. Finetune with cross-sentence constraint. (change the model path in the config file: MODEL/CHECKPOINT)
```
python train.py \
         --cfg ./configs/YouwikiHow/16x16_MIL_CSv2_L01D01_continue_HardSS_K7D05_try2.yaml \
         --tag MIL_16x16_CSv2_L01D01_continue_HardSS_K7D05_try2_coconut --verbose --online_feat \
         --tensorboard

```


## Pretrained Weights
These [Provided Weights](https://drive.google.com/drive/folders/1gG31AjpIqKjfLmRH6trga8o1KBvKPpqt?usp=sharing) can be saved into the directory ``checkopoints``, such as:
```
WSAG/
|   +-- checkpoints/
|       +-- wikihow_grounding/ # previded weights
|       +-- YouwikiHow/ # new saved weights
```

## Citations
```
@inproceedings{chen2022weakly,
  title={Weakly-Supervised Temporal Article Grounding},
  author={Chen, Long and Niu, Yulei and Chen, Brian and Lin, Xudong and Han, Guangxing and Thomas, Christopher and Ayyubi, Hammad and Ji, Heng and Chang, Shih-Fu},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP), 2022},
  year={2022}
}
```