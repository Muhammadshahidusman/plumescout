"""PlumeScout demo: Sentinel-2 methane plume detections over Gulf oil and gas sites.

Precomputed assets only (app/assets, built by `uv run python -m plumescout.demo`): no Earth
Engine calls and no credentials. Detections are coordinates and dates; no emission rates
are estimated and no operator is named.

Run locally: uv run python app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

try:  # ZeroGPU Spaces refuse to start without one @spaces.GPU function; absent locally.
    import spaces

    @spaces.GPU
    def _zerogpu_stub() -> None:
        """Never called: every panel is precomputed, the demo needs no GPU."""

except ImportError:
    pass

ASSETS = Path(__file__).resolve().parent / "assets"
INDEX = json.loads((ASSETS / "index.json").read_text())
SITES = {s["label"]: s for s in INDEX["sites"]}
PANEL_TITLES = {
    "rgb": "True colour (B4, B3, B2)",
    "dr": "MBMP B12/B11 enhancement (blue = negative = possible CH4)",
    "pred": "U-Net probability (cyan = detection, dashed grey = flare-masked)",
    "ref": "Reference plumes within 1 day (green = EMIT, yellow = Carbon Mapper)",
}


def _date_choices(label: str) -> list[str]:
    out = []
    for s in SITES[label]["scenes"]:
        tag = []
        if s["n_det"]:
            tag.append("detection")
        if s["has_reference"]:
            tag.append("reference plume")
        out.append(f"{s['date']} {s['time_utc']} UTC" + (f"  [{', '.join(tag)}]" if tag else ""))
    return out


def _metrics_md() -> str:
    m = INDEX["metrics"]
    rows = []
    for key, name in [
        ("unet", "U-Net (trained on CH4Net, Turkmenistan)"),
        ("mbmp", "MBMP baseline"),
    ]:
        e, ci = m[key]["eligible"], m[key]["ci95_site_bootstrap"]
        rows.append(
            f"| {name} | {e['plume_recall']:.2f} [{ci['plume_recall'][0]:.2f}, "
            f"{ci['plume_recall'][1]:.2f}] | {e['precision']:.2f} [{ci['precision'][0]:.2f}, "
            f"{ci['precision'][1]:.2f}] | {e['fp_per_100_clear_tn']:.0f} |"
        )
    e = m["unet"]["eligible"]
    return (
        f"**Gulf evaluation** ({e['n_sites']} sites, {e['n_ref_plumes']} reference plumes "
        f"with a clear scene, {e['n_clear_tn_scenes']} clear EMIT no-plume scenes; "
        "95% CI by site bootstrap). Sites within 2 km of a training location are excluded.\n\n"
        "| Method | Plume recall | Precision | False detections per 100 clear no-plume scenes |\n"
        "|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "On these unseen sites the U-Net does not beat the classic MBMP baseline. Treat every "
        "detection as a candidate for review, not a confirmed emission."
    )


def show(label: str, date_choice: str):
    site = SITES[label]
    date = date_choice.split(" ")[0]
    scene = next(s for s in site["scenes"] if s["date"] == date)
    imgs = [str(ASSETS / scene["files"][p]) for p in ("rgb", "dr", "pred", "ref")]
    head = f"**{label}** ({site['lat']:.4f} N, {site['lon']:.4f} E), {date} {scene['time_utc']} UTC"
    return (*imgs, f"{head}\n\n{scene['verdict']}")


def on_site(label: str):
    choices = _date_choices(label)
    # open on a date with both a detection and a reference plume, else any tagged date
    both = [c for c in choices if "detection" in c and "reference" in c]
    first = both[0] if both else next((c for c in choices if "[" in c), choices[0])
    return gr.update(choices=choices, value=first), *show(label, first)


with gr.Blocks(title="PlumeScout") as demo:
    gr.Markdown(
        "# PlumeScout\nMethane plume candidates from Sentinel-2 L1C over oil and gas sites in "
        "Kuwait, southern Iraq and Oman, checked against NASA EMIT V002 and Carbon Mapper. Pick "
        "a site and a date. Dates tagged [detection] or [reference plume] are the interesting ones."
    )
    labels = list(SITES)
    with gr.Row():
        site_dd = gr.Dropdown(labels, value=labels[0], label="Site")
        date_dd = gr.Dropdown(_date_choices(labels[0]), label="Sentinel-2 overpass (UTC)")
    verdict_md = gr.Markdown()
    with gr.Row():
        panels = [
            gr.Image(label=PANEL_TITLES[p], type="filepath", height=300) for p in ("rgb", "dr")
        ]
    with gr.Row():
        panels += [
            gr.Image(label=PANEL_TITLES[p], type="filepath", height=300) for p in ("pred", "ref")
        ]
    gr.Markdown(_metrics_md())
    gr.Markdown(f"<small>{INDEX['attribution']}</small>")

    site_dd.change(on_site, site_dd, [date_dd, *panels, verdict_md])
    date_dd.change(show, [site_dd, date_dd], [*panels, verdict_md])
    demo.load(on_site, site_dd, [date_dd, *panels, verdict_md])

if __name__ == "__main__":
    demo.launch()
