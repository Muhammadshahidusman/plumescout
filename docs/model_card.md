# Model card: PlumeScout U-Net (CH4Net reproduction)

**Status: not published.** CH4Net is released under CC BY-NC-ND 4.0. Trained weights may count as
a derivative ("ND"), so publishing them needs the CH4Net authors' permission or a licence review.

## Model
- U-Net, ResNet-34 encoder (ImageNet init), 13 input channels: 12 Sentinel-2 bands in CH4Net's
  uint8 scale plus the CH4Net MBMP field decoded to [0, 1]. Output: per-pixel plume probability.
- Checkpoint `paper_allbands_30ep/best.ckpt` (epoch 1 of 30, selected on CH4Net val scene balanced
  accuracy). Loss: BCE + Dice. Augmentation: flips and 90 degree rotations only.

## Training data
CH4Net (Vaughan et al. 2024, AMT 17, 2583; HF `av555/ch4net`): 23 Turkmenistan sites, 2017 to 2020
train (8,255 images), val 255 images. No Gulf data were used for training or tuning.

## Evaluation (all numbers with counts; CIs 95%)
| Setting | Data | Result |
|---|---|---|
| CH4Net test 2021, paper split (same sites as training) | 2,473 scenes, 275 positive, 23 sites | scene recall 0.884 [0.846, 0.919], precision 0.214, FPR 0.406, balanced acc. 0.739 (scene bootstrap) |
| CH4Net, site-disjoint (separate run, 20 epochs) | 583 scenes, 48 positive, 7 unseen sites | balanced acc. 0.651 [0.411, 0.763] at threshold 0.25 (site bootstrap) |
| CH4Net, 5-fold grouped site CV (separate runs, 10 epochs each) | 2,473 scenes, 275 positive, 23 sites, each tested once | scene recall 0.542 [0.321, 0.703], precision 0.258, FPR 0.195, balanced acc. 0.673 [0.567, 0.748] at each fold's validation threshold (site bootstrap) |
| Gulf, 16 eligible sites (this checkpoint, threshold 0.2, >= 116 px) | 59 reference plumes, 143 clear EMIT no-plume scenes | plume recall 0.220 [0.000, 0.414], precision 0.190 [0.000, 0.462], 44.8 FP per 100 clear no-plume scenes (site bootstrap) |
| MBMP baseline, same Gulf scenes | same | plume recall 0.271 [0.125, 0.412], precision 0.188 [0.071, 0.312] |

## Intended use and limits
- Research and portfolio use only (non-commercial). Detections are coordinates and dates of
  plume-like signals for human review; the model does not estimate emission rates and must not be
  used to rank or blame operators.
- Inputs outside Turkmenistan require mapping L1C reflectance to CH4Net's uint8 scale (fitted on
  Turkmenistan) and an MBMP stretch; CH4Net's exact MBMP preprocessing is not reproduced.
- The model relies almost entirely on the MBMP channel, so MBMP artefacts (surface change between
  dates, linear features, cloud edges) become false detections.
- Performance varies strongly by site and is weak on unseen sites.
