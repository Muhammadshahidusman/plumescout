---
title: PlumeScout
emoji: 🛰️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: other
short_description: Sentinel-2 methane plume candidates over Gulf oil fields
---

# PlumeScout demo

Methane plume candidates from Sentinel-2 L1C over 19 oil and gas sites in Kuwait, southern Iraq
and Oman, compared with NASA EMIT V002 and Carbon Mapper reference plumes. All panels are
precomputed: the Space makes no Earth Engine calls and holds no credentials.

On 16 unseen Gulf sites the U-Net (trained on CH4Net, Turkmenistan) reaches plume recall 0.22
[0.00, 0.41] and precision 0.19 [0.00, 0.46], no better than the classic MBMP baseline (0.27, 0.19).
Detections are candidates for review. No emission rates are estimated and no operator is named.

Code, method and full results: [github.com/Muhammadshahidusman/plumescout](https://github.com/Muhammadshahidusman/plumescout).

## Licence and attribution

Non-commercial use only. Contains modified Copernicus Sentinel-2 data (ESA). NASA EMIT L2B CH4 V002
(LP DAAC). Source: Carbon Mapper; Carbon Mapper plume outlines are shown under the Carbon Mapper Terms
of Use (13 January 2026) and any reuse must be non-commercial and attributed on the same terms. NASA
FIRMS VIIRS. Model trained on CH4Net (Vaughan et al. 2024, CC BY-NC-ND 4.0); model weights are not
published.
