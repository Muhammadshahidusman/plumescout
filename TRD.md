# TRD: PlumeScout

**Owner:** Muhammad Shahid Usman
**Status:** Draft v0.1
**Date:** 2026-11-04
**Companion doc:** PRD.md

---

## 1. System overview

```
             +---------------------------+
             |  Site list (sites.geojson)|
             +-------------+-------------+
                           |
      +--------------------+---------------------+
      |                    |                     |
 Sentinel-2 L1C       EMIT L2B V002         Carbon Mapper
 (GEE / CDSE)         (earthaccess)         (portal export)
      |                    |                     |
      v                    v                     v
 chips/ (200x200)     ref/emit/*.geojson    ref/cm/*.geojson
      |
      +--> MBMP baseline (physics)
      |
      +--> U-Net (PyTorch) --> plume masks --> polygons
                                     |
                         VIIRS flare mask (FIRMS)
                                     |
                                     v
                     Matcher (±1 day, 1 km) vs EMIT / CM
                                     |
                                     v
                 metrics/ + figures/ + demo (Gradio, HF Space)
```

Batch pipeline, run locally on Windows with a CUDA GPU. No server, no database in v1. Files on disk, tracked by Parquet tables.

## 2. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Env | `uv`, Python 3.11 | Same as Kinara; fast, reproducible |
| Raster I/O | `rasterio`, `rioxarray`, `xarray` | Standard |
| Vector | `geopandas`, `shapely` 2.x | Standard |
| Imagery access | `earthengine-api`, `geemap` | Sentinel-2 L1C archive server-side |
| NASA data | `earthaccess` | EMIT V002 (not in GEE at V002) |
| DL | `torch`, `segmentation-models-pytorch`, `lightning` | Fast U-Net setup |
| Labels | `marss2l` package, CH4Net dataset | Public plume masks |
| Tables | `pandas`, `pyarrow` (Parquet) | Scene and detection logs |
| Demo | `gradio` on Hugging Face Spaces | Free hosting, recruiter-friendly |
| Viz | `matplotlib`, `folium` | Figures and maps |
| Tests | `pytest` | Pipeline units |
| Lint | `ruff` | One tool |

## 3. Data sources

| # | Dataset | ID / location | Access | Licence | Role |
|---|---|---|---|---|---|
| D1 | CH4Net plume dataset | Linked from Vaughan et al., AMT 17, 2583 (2024) | Free download | CC BY 4.0 (paper) | Training labels v1 (925 masks, 23 Turkmenistan sites) |
| D2 | MARS-S2L | HF `UNEP-IMEO/MARS-S2L`, pip `marss2l` | Free, ~100 GB | CC non-commercial SA | Bigger training set (~87k images, 5,600+ plumes) |
| D3 | Sentinel-2 L1C | GEE `COPERNICUS/S2_HARMONIZED` | Free (GEE account) | Copernicus open | Inference imagery for Gulf sites |
| D4 | EMIT L2B CH4 Plume Complexes V002 | LP DAAC `EMITL2BCH4PLM` v002 | Free (Earthdata login) | NASA open | Validation reference |
| D5 | EMIT L2B CH4 Enhancement V002 | LP DAAC `EMITL2BCH4ENH` v002 | Free (Earthdata login) | NASA open | True negatives, visual QA |
| D6 | Carbon Mapper plumes | data.carbonmapper.org | Free account | Non-commercial | Second validation reference |
| D7 | VIIRS active fire 375 m | NASA FIRMS (VNP14IMG / VJ114IMG) | Free MAP_KEY | NASA open | Flare masking |
| D8 | Facility points | OSM (`man_made=petroleum_well`, `industrial=oil`), Global Energy Monitor | Free | ODbL / CC BY | Site selection context |

**Rules:**
- Use **L1C, not L2A.** Atmospheric correction can distort the B12 methane signal.
- Use **EMIT V002 only.** V001 was decommissioned on 2026-03-26. GEE still points to V001, so pull V002 via `earthaccess`.
- Record licence in every derived file's metadata.

## 4. Study regions

| Region | Candidate fields | Role |
|---|---|---|
| Turkmenistan (Hazar / Galkynysh area) | CH4Net sites | Training, baseline reproduction |
| Kuwait | Burgan area | Gulf test |
| Southern Iraq | Rumaila, West Qurna, Zubair (Basra) | Gulf test |
| Oman | Interior oil fields | Gulf test |
| Permian Basin, USA | Optional | Extra validation only |

Final site list (15 to 20 points) is chosen from EMIT V002 and Carbon Mapper detections inside these AOIs, stored in `data/sites/sites.geojson` with fields: `site_id`, `region`, `lon`, `lat`, `source` (emit / cm / osm), `in_train_overlap` (bool).

## 5. Pipeline stages

### 5.1 Site selection (`src/plumescout/sites.py`)
- Query EMIT V002 plume GeoJSONs and Carbon Mapper exports within AOIs.
- Cluster plumes within 1 km into one site (DBSCAN, haversine).
- Flag any site within 2 km of a MARS-S2L or CH4Net training location: `in_train_overlap = True`. Those sites are excluded from Gulf test metrics.

### 5.2 Imagery download (`src/plumescout/s2_download.py`)
- For each site and date, export a 200x200 px chip centred on the site, all 13 bands resampled to 10 m, UTM CRS of the site.
- Also export a reference chip: the most recent clear scene from the prior 30 days (needed for MBMP).
- Store cloud fraction from `QA60` / s2cloudless in the scene table.
- Output: `data/chips/{site_id}/{YYYYMMDD}.tif`

### 5.3 MBMP baseline (`src/plumescout/mbmp.py`)
- Multi-band single-pass: `ΔR = (c · R12 − R11) / R11`, with `c` fit by least squares of R11 on R12 over the chip.
- Multi-band multi-pass: subtract the reference scene's ΔR.
- Threshold to a binary mask for baseline metrics.

### 5.4 Model (`src/plumescout/model.py`)
- Architecture: U-Net, ResNet-34 encoder (ImageNet weights, first conv adapted to N input channels).
- Input channels v1: B11, B12, B8A, B4, B3, B2, plus MBMP ΔR = 7 channels.
- Loss: BCE + Dice (plumes are under 5% of pixels).
- Optimizer: AdamW, lr 1e-4, cosine schedule, 50 epochs, batch 16.
- Augmentations: flips and 90° rotations only. No colour jitter (it corrupts the spectral signal).

### 5.5 Training splits
- **Split by site, never by image.** Random image splits leak site background.
- CH4Net: follow paper (train 2017 to 2020, test 2021) for reproduction.
- MARS-S2L: use the package's own train/test split.
- Gulf sites: test only. Never trained on.

### 5.6 Inference and post-processing (`src/plumescout/infer.py`)
- Sigmoid threshold 0.5 (tuned on validation set, then frozen).
- Remove components under 10 pixels.
- Remove components overlapping a VIIRS fire detection within ±12 h and 750 m (flare mask).
- Vectorise to polygons, write `data/detections/{site_id}.geojson`.

### 5.7 Matching (`src/plumescout/match.py`)
- A Sentinel-2 detection is a **true positive** if an EMIT or Carbon Mapper plume exists within 1 km and ±1 day.
- A **false negative** is a reference plume on a clear (<20% cloud) S2 chip with no detection.
- A **false positive** is a detection with no reference plume, on a date where EMIT observed the site and found nothing (uses D5 enhancement granules for true negatives).
- Detections on dates with no reference coverage are logged as `unverified`, not counted.

## 6. Data tables (Parquet)

**scenes.parquet**
| col | type |
|---|---|
| scene_id | str |
| site_id | str |
| date | date |
| s2_product | str |
| cloud_frac | float |
| ref_scene_date | date |
| chip_path | str |

**detections.parquet**
| col | type |
|---|---|
| det_id | str |
| scene_id | str |
| area_px | int |
| max_prob | float |
| flare_masked | bool |
| geometry_wkt | str |
| match_source | str (emit / cm / none) |
| match_status | str (tp / fp / unverified) |

**references.parquet**
| col | type |
|---|---|
| ref_id | str |
| source | str |
| date | datetime |
| site_id | str |
| geometry_wkt | str |
| emission_rate_kgh | float (stored, never claimed as ours) |

## 7. Evaluation

| Metric | Level | Notes |
|---|---|---|
| Recall, precision, F1 | Plume (object) | Primary |
| IoU | Pixel | Secondary, only vs CH4Net / MARS-S2L masks |
| FP per 100 clear scenes | Site | Report per site |
| FP by cause | Site | Flare, bright surface, cloud edge, unknown |

Always report: number of scenes, number of reference plumes, number of sites, and the overlap-excluded count. Bootstrap 95% CIs by site (1,000 resamples).

## 8. Demo (`app/`)
- Gradio app, deployed to HF Space `Muhammadshahidusman/plumescout`.
- Precomputed chips and predictions for demo sites (no live GEE calls; keeps it fast and avoids credentials on the Space).
- Panels: RGB, ΔR enhancement, predicted mask, EMIT outline, verdict text.
- Size budget: under 500 MB of demo assets.

## 9. Repo structure

```
plumescout/
├── CLAUDE.md
├── PRD.md
├── TRD.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── sites.yaml
│   └── train.yaml
├── data/              # gitignored
│   ├── raw/
│   ├── chips/
│   ├── ref/
│   ├── sites/
│   └── detections/
├── models/            # gitignored, weights pushed to HF Hub
├── notebooks/
│   ├── 01_mbmp_turkmenistan.ipynb
│   ├── 02_train_ch4net.ipynb
│   ├── 03_gulf_sites.ipynb
│   └── 04_evaluation.ipynb
├── src/plumescout/
│   ├── sites.py
│   ├── s2_download.py
│   ├── emit.py
│   ├── carbonmapper.py
│   ├── firms.py
│   ├── mbmp.py
│   ├── model.py
│   ├── train.py
│   ├── infer.py
│   └── match.py
├── app/
│   └── app.py
├── tests/
└── figures/
```

## 10. Secrets and config
- `.env` (gitignored): `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, `FIRMS_MAP_KEY`, `HF_TOKEN`, `GEE_PROJECT`.
- Never commit credentials or Carbon Mapper raw exports.

## 11. Compute
- Local: Windows, CUDA GPU (existing `D:\torchgpu` env as reference).
- Training budget v1: under 4 GPU-hours on CH4Net.
- Fallback: Kaggle or Colab GPU.
- Disk: ~5 GB for v1 (CH4Net + Gulf chips). MARS-S2L full set needs ~100 GB; subset by region if used.

## 12. Testing
- `test_mbmp.py`: synthetic chip with injected B12 absorption returns negative ΔR (Varon et al. 2021 sign convention: methane absorbs in B12, so plume pixels have ΔR < 0).
- `test_match.py`: known TP / FP / unverified cases.
- `test_split.py`: asserts no site appears in both train and test.
- `test_flare_mask.py`: detection on a FIRMS point is removed.

## 13. Known limitations (state in README)
- Sentinel-2 only sees large plumes (super-emitters).
- Bright desert surfaces and flares cause false positives.
- Validation is limited by EMIT and Carbon Mapper overpass coverage.
- No emission rate quantification.
