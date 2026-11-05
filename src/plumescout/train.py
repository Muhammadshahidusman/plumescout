"""Train the U-Net on CH4Net and evaluate plume- and scene-level metrics.

CLI:
    uv run python -m plumescout.train --config configs/train.yaml [--epochs 2] [--run-name smoke]

Writes models/{run}/best.ckpt, models/{run}/metrics.json and per-scene counts
models/{run}/test_scenes.parquet. The probability threshold is chosen on val
(plume-level F1) and then frozen for test.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import lightning as L  # noqa: N812 - Lightning convention
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from plumescout import metrics
from plumescout.config import load_yaml, resolve_path
from plumescout.dataset import CH4NetDataset
from plumescout.model import BceDiceLoss, build_unet

logger = logging.getLogger(__name__)


def ch4net_items(root: Path) -> pd.DataFrame:
    """All CH4Net images as rows (folder, image_id)."""
    rows = [
        {"folder": folder, "image_id": int(f.stem)}
        for folder in ("train", "val", "test")
        for f in (root / folder / "s2").glob("*.npy")
    ]
    return pd.DataFrame(rows).sort_values(["folder", "image_id"]).reset_index(drop=True)


def make_split(items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach a `split` column according to cfg['data']['split']."""
    from plumescout import split as sp

    mode = cfg["data"]["split"]
    if mode == "paper":
        return sp.paper_split(items)
    if mode == "site_year":
        idx = pd.read_parquet(resolve_path("data/ch4net_index_all.parquet"))
        items = items.merge(
            idx[["folder", "image_id", "site_id"]], on=["folder", "image_id"], how="left"
        )
        if items.site_id.isna().any():
            raise ValueError(f"{int(items.site_id.isna().sum())} images have no site_id")
        sites = pd.read_csv(resolve_path(load_yaml("sites.yaml")["ch4net_sites_csv"]))
        groups = sp.site_groups(sites, cfg["data"]["site_group_eps_m"])
        return sp.site_year_split(
            items,
            groups,
            cfg["data"]["test_frac_groups"],
            cfg["data"]["val_frac_groups"],
            cfg["seed"],
        )
    raise ValueError(f"Unknown split mode {mode!r}")


class PlumeModule(L.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        m = cfg["model"]
        self.net = build_unet(m["encoder"], m["encoder_weights"], m["in_channels"], m["classes"])
        self.loss = BceDiceLoss(cfg["loss"]["bce_weight"], cfg["loss"]["dice_weight"])
        self.cfg = cfg
        self._val_rows: dict[float, list[dict]] = {}
        self._val_scene: dict[float, list[bool]] = {}
        self._val_pos: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y, valid, _ = batch
        logits = self(x)
        v = valid[:, None].float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=v)
        self.log("val_bce", loss, prog_bar=True, on_epoch=True, on_step=False)
        inf = self.cfg["inference"]
        probs = [
            (0, p[0], t[0] > 0.5, m)
            for p, t, m in zip(
                torch.sigmoid(logits).cpu().numpy(),
                y.cpu().numpy(),
                valid.cpu().numpy(),
                strict=True,
            )
        ]
        for t, rows in evaluate(probs, inf["threshold_grid"], inf["min_component_px"]).items():
            self._val_rows.setdefault(t, []).extend(rows)
        for t, flags in scene_flags(probs, inf["threshold_grid"], inf["scene_min_px"]).items():
            self._val_scene.setdefault(t, []).extend(flags)
        self._val_pos.extend(bool(y.any()) for _, _, y, _ in probs)

    def on_validation_epoch_end(self):
        """Log val scene balanced accuracy (paper rule, best grid threshold), which drives
        checkpoint selection, and val plume F1 for reference."""
        if not self._val_rows:
            return
        scores = {t: metrics.summarise(r) for t, r in self._val_rows.items()}
        thr = metrics.best_threshold(scores)
        self.log("val_plume_f1", float(np.nan_to_num(scores[thr]["plume_f1"])), prog_bar=True)
        sc = {t: metrics.scene_scores(self._val_pos, f) for t, f in self._val_scene.items()}
        st = metrics.best_threshold(sc, key="balanced_accuracy")
        self.log("val_scene_bacc", float(np.nan_to_num(sc[st]["balanced_accuracy"])), prog_bar=True)
        self.log("val_scene_thr", float(st))
        self._val_rows, self._val_scene, self._val_pos = {}, {}, []

    def configure_optimizers(self):
        o = self.cfg["optim"]
        opt = torch.optim.AdamW(self.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self.trainer.max_epochs)
        )
        return {"optimizer": opt, "lr_scheduler": sched}


class _EpochSetter(L.Callback):
    """Give the training dataset a new augmentation seed each epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        trainer.train_dataloader.dataset.epoch = trainer.current_epoch


@torch.no_grad()
def predict_probs(model: PlumeModule, loader: DataLoader, device: torch.device):
    """Yield (index, prob map, label, valid) per image, as numpy."""
    model.eval().to(device)
    for x, y, valid, idx in loader:
        p = torch.sigmoid(model(x.to(device))).cpu().numpy()[:, 0]
        for k in range(len(idx)):
            yield int(idx[k]), p[k], y[k, 0].numpy() > 0.5, valid[k].numpy()


def evaluate(probs: list, thresholds: list[float], min_px: int) -> dict[float, list[dict]]:
    """Per-scene plume counts for each threshold."""
    out = {t: [] for t in thresholds}
    for _, p, y, v in probs:
        for t in thresholds:
            out[t].append(metrics.plume_counts(p >= t, y, min_px, valid=v))
    return out


def scene_flags(probs: list, thresholds: list[float], min_px: int) -> dict[float, list[bool]]:
    """Paper scene rule flag per scene for each threshold."""
    return {t: [metrics.scene_flag(p, t, min_px, v) for _, p, _, v in probs] for t in thresholds}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train U-Net on CH4Net.")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override optim.epochs")
    ap.add_argument("--limit-train-batches", type=float, default=1.0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--split", choices=["paper", "site_year"], default=None, help="override")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_yaml(args.config)
    n_in = len(cfg["data"]["input_bands"]) + int(cfg["data"]["add_mbmp_channel"])
    if cfg["model"]["in_channels"] != n_in:
        raise ValueError(f"model.in_channels={cfg['model']['in_channels']} but data gives {n_in}")
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    if args.split is not None:
        cfg["data"]["split"] = args.split
    L.seed_everything(cfg["seed"], workers=True)
    run = args.run_name or f"{cfg['data']['split']}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = resolve_path(cfg["trainer"]["output_dir"]) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_path(cfg["data"]["root"])
    items = make_split(ch4net_items(root), cfg)
    counts = items.split.value_counts().to_dict()
    logger.info("split=%s counts=%s", cfg["data"]["split"], counts)
    d, tr = cfg["data"], cfg["trainer"]
    mk = lambda part, train: CH4NetDataset(  # noqa: E731
        root,
        items[items.split == part],
        d["input_bands"],
        train,
        crop=d["train_crop_px"],
        pad_hw=tuple(d["eval_pad_hw"]),
        seed=cfg["seed"],
    )
    ds_train, ds_val, ds_test = mk("train", True), mk("val", False), mk("test", False)
    bs, nw = cfg["optim"]["batch_size"], tr["num_workers"]
    dl_train = DataLoader(
        ds_train, bs, shuffle=True, num_workers=nw, persistent_workers=nw > 0, drop_last=True
    )
    dl_val = DataLoader(ds_val, bs, num_workers=0)  # keep RAM low: workers only for training
    dl_test = DataLoader(ds_test, bs, num_workers=0)

    model = PlumeModule(cfg)
    ckpt = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=out_dir, filename="best", monitor="val_scene_bacc", mode="max"
    )
    # save_last=True on the monitored callback only rewrote last.ckpt when best improved
    # (seen 2026-11-05), so keep the latest epoch with a separate unmonitored callback.
    last = L.pytorch.callbacks.ModelCheckpoint(dirpath=out_dir, filename="last_epoch", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=cfg["optim"]["epochs"],
        accelerator=tr["accelerator"],
        precision=tr["precision"],
        callbacks=[ckpt, last, _EpochSetter()],
        limit_train_batches=args.limit_train_batches,
        default_root_dir=out_dir,
        log_every_n_steps=20,
        deterministic=False,
    )
    t0 = time.time()
    trainer.fit(model, dl_train, dl_val)
    train_s = time.time() - t0
    best_epoch = int(
        torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["epoch"]
    )
    logger.info("best checkpoint: epoch %d, val_scene_bacc %.3f", best_epoch, ckpt.best_model_score)
    best = PlumeModule.load_from_checkpoint(ckpt.best_model_path, cfg=cfg)

    inf = cfg["inference"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_probs = list(predict_probs(best, DataLoader(ds_val, bs), dev))
    val_counts = evaluate(val_probs, inf["threshold_grid"], inf["min_component_px"])
    val_pos = [bool(y.any()) for _, _, y, _ in val_probs]
    val_scene = {
        t: metrics.scene_scores(val_pos, f)
        for t, f in scene_flags(val_probs, inf["threshold_grid"], inf["scene_min_px"]).items()
    }
    scene_thr = metrics.best_threshold(val_scene, key="balanced_accuracy")
    del val_probs
    val_scores = {t: metrics.summarise(r) for t, r in val_counts.items()}
    thr = metrics.best_threshold(val_scores)
    logger.info(
        "threshold chosen on val: %.2f (val plume F1 %.3f)", thr, val_scores[thr]["plume_f1"]
    )

    test_probs = list(predict_probs(best, dl_test, dev))
    test_rows = evaluate(test_probs, [thr], inf["min_component_px"])[thr]
    test = metrics.summarise(test_rows)
    ious = [metrics.iou((p >= thr) & v, y & v) for _, p, y, v in test_probs if y.any()]
    test["pixel_iou_positive_scenes_mean"] = float(np.nanmean(ious)) if ious else float("nan")
    test_pos = [bool(y.any()) for _, _, y, _ in test_probs]
    sel_test = items[items.split == "test"].reset_index(drop=True)
    test_sites = sel_test.site_id.to_numpy() if "site_id" in sel_test else None
    paper_thr = inf["paper_scene_threshold"]
    scene_rule = {
        f"thr_{t}": {
            **metrics.scene_scores(test_pos, f),
            "ci95_scene_bootstrap": metrics.bootstrap_scene_ci(test_pos, f, seed=cfg["seed"]),
            **(
                {
                    "ci95_site_bootstrap": metrics.bootstrap_scene_ci(
                        test_pos, f, seed=cfg["seed"], groups=test_sites
                    )
                }
                if test_sites is not None
                else {}
            ),
        }
        for t, f in scene_flags(test_probs, [scene_thr, paper_thr], inf["scene_min_px"]).items()
    }
    logger.info("TEST scene rule %s", json.dumps(scene_rule, default=float))

    sel = items[items.split == "test"].reset_index(drop=True)
    per_scene = pd.concat([sel, pd.DataFrame(test_rows)], axis=1)
    per_scene.to_parquet(out_dir / "test_scenes.parquet", index=False)
    result = {
        "run": run,
        "split": cfg["data"]["split"],
        "reproduction_only": cfg["data"]["split"] == "paper",
        "test_sites": sorted(set(sel.site_id), key=lambda x: int(x[1:]))
        if "site_id" in sel
        else None,
        "split_counts": counts,
        "epochs": cfg["optim"]["epochs"],
        "limit_train_batches": args.limit_train_batches,
        "train_seconds": round(train_s, 1),
        "best_ckpt": Path(ckpt.best_model_path).name,
        "best_epoch": best_epoch,
        "checkpoint_monitor": "val_scene_bacc (paper scene rule, max over threshold grid)",
        "threshold": thr,
        "val": val_scores[thr],
        "test": test,
        "scene_rule": {
            "definition": f"scene flagged if an 8-connected region with p >= thr has "
            f">= {inf['scene_min_px']} px (CH4Net paper: > 115 px)",
            "threshold_from_val_bacc": scene_thr,
            "val": val_scene[scene_thr],
            "test": scene_rule,
        },
        "data_licence": "CH4Net (av555/ch4net) CC BY-NC-ND 4.0",
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    logger.info("TEST %s", json.dumps(test, default=float))


if __name__ == "__main__":
    main()
