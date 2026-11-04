# PRD: PlumeScout

*Methane plume detection over Gulf oil fields from Sentinel-2.*

**Owner:** Muhammad Shahid Usman
**Status:** Draft v0.1
**Date:** 2026-11-04
**Type:** Portfolio project (non-commercial)

---

## 1. One-line pitch

An open pipeline that flags methane super-emitter plumes over Middle East oil and gas fields from free Sentinel-2 imagery, validated against NASA EMIT and Carbon Mapper detections.

## 2. Why this exists

Methane from oil and gas is one of the fastest climate wins available, and Gulf operators (Saudi Aramco, ADNOC, KOC, Basra Oil Company) are under growing pressure to prove they're monitoring it. Satellite methane detection is already a real industry: Kayrros, GHGSat, Carbon Mapper and UNEP IMEO all sell or publish it.

The gap this project fills is personal, not market-wide. My portfolio shows land cover, change detection and flood work. Nothing in it proves I can do atmospheric remote sensing, work with hyperspectral reference data, or handle a hard, noisy detection problem honestly. This project closes that gap and speaks directly to Gulf energy employers.

## 3. Target audience

| Audience | What they need to see |
|---|---|
| Gulf energy hiring managers (Aramco, ADNOC, KOC, QatarEnergy) | I understand methane remote sensing and their operating terrain |
| GeoAI / climate-tech recruiters (Kayrros, GHGSat, Orbital Sidekick, Planet) | Clean ML pipeline, honest validation, reproducible code |
| Technical interviewers | Correct physics (B11/B12), no data leakage, real false-positive analysis |

## 4. Goals (v1)

1. Detect methane plumes in Sentinel-2 L1C imagery with a segmentation model trained on public labels.
2. Apply it to 15 to 20 oil and gas sites in Kuwait, southern Iraq and Oman.
3. Validate each detection against NASA EMIT V002 and Carbon Mapper, within ±1 day.
4. Report per-site recall, precision and false positives, and explain the failures.
5. Ship a public demo (Hugging Face Space) and a GitHub repo with a strong README.

## 5. Non-goals (v1)

- **No emission rate estimates (kg/h).** Sentinel-2 detection limits sit around 1,000 to 2,000 kg/h on uniform surfaces and above 5,000 kg/h on mixed ones. Bad quantification kills credibility. Detection only.
- **No polluter leaderboard.** Results are framed as "detections at coordinates, cross-checked with EMIT," never "worst operators."
- **No commercial use.** MARS-S2L and Carbon Mapper data are non-commercial licences.
- **No real-time alerting.** Batch processing only.
- **No hyperspectral retrievals of my own.** EMIT is used as reference, not reprocessed.

## 6. Users and core flows

**Flow A: Recruiter browsing the demo**
1. Opens Hugging Face Space.
2. Picks a site from a dropdown (e.g. "Southern Iraq, site 04").
3. Picks a date.
4. Sees: RGB image, B12/B11 enhancement image, predicted plume mask, EMIT outline if available.
5. Reads a 2-line verdict: "Plume detected. EMIT confirmed on same day."

**Flow B: Technical reviewer reading the repo**
1. Reads README: problem, data, method, results table, limitations.
2. Runs `uv sync` and one notebook to reproduce a site result.
3. Checks the evaluation notebook for split design and leakage handling.

## 7. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| F1 | Download Sentinel-2 L1C chips (200x200 px) for a site and date range | Must |
| F2 | Compute classic MBMP enhancement (Varon et al. 2021) as a baseline | Must |
| F3 | Train a U-Net segmentation model on CH4Net or MARS-S2L labels | Must |
| F4 | Run inference on Gulf sites and export plume polygons (GeoJSON) | Must |
| F5 | Match detections to EMIT V002 plumes within ±1 day and 1 km | Must |
| F6 | Match detections to Carbon Mapper plumes the same way | Should |
| F7 | Mask gas flares using VIIRS active fire (FIRMS) | Must |
| F8 | Produce per-site metrics table and failure gallery | Must |
| F9 | Gradio demo on Hugging Face Spaces | Must |
| F10 | Model weights published on Hugging Face Hub | Should |
| F11 | Landfill follow-up on Pakistani sites (e.g. Karachi) | Could (v2) |

## 8. Success metrics

| Metric | Target |
|---|---|
| Baseline reproduction on CH4Net test set | Recall within 10 points of the paper's 84% |
| Gulf sites covered | 15 or more |
| EMIT-matched scene pairs for validation | 30 or more |
| Plume-level recall on Gulf EMIT matches | Reported honestly; no fixed target |
| False positives per 100 clear scenes | Reported per site |
| Demo load time | Under 10 s for a cached site |
| README | Results table, 4 figures, limitations section |
| LinkedIn post | Published with 3 to 4 before/after images |

The honest number matters more than a high number. A 55% recall with a clear explanation beats an unexplained 90%.

## 9. Milestones

| Week | Deliverable |
|---|---|
| 1 | MBMP baseline working on 3 Turkmenistan super-emitter sites |
| 2 | U-Net trained on CH4Net labels; baseline metrics reproduced |
| 3 | Gulf site list (15 to 20) and matched Sentinel-2 / EMIT / Carbon Mapper scene table |
| 4 | Gulf inference, flare masking, per-site evaluation |
| 5 | Gradio demo on HF Spaces; weights on HF Hub |
| 6 | README, figures, LinkedIn post |

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Few EMIT overpasses on Gulf sites | Thin validation set | Widen AOI; add Carbon Mapper; accept ±1 day |
| Gulf sites already in MARS-S2L train split | Inflated metrics | Check site overlap before training; hold out overlaps |
| Flares and bright sand trigger false positives | Low precision | VIIRS flare mask; report FP separately by cause |
| 100 GB MARS-S2L too big for local disk | Blocks training | Start with CH4Net (small); subset MARS-S2L by region |
| GPU limits | Slow training | Small chips (200x200), ResNet-34 encoder, Colab or Kaggle GPU as fallback |
| Sensitive framing with target employers | Reputation | No operator names on detections; neutral language |

## 11. Open questions

1. CH4Net labels or MARS-S2L subset for v1 training? (Default: CH4Net first.)
2. Include Permian Basin as an extra validation region? (Default: yes, validation only.)
3. Tagline for README and portfolio: "Methane plume detection over Gulf oil fields from Sentinel-2." (Proposed.)
