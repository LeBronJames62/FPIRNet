"""Evaluate SAVED images: paired PSNR/SSIM/LPIPS, no-reference IQA, or fixed-ROI OCR.

This file never imports FPIRNet, restores images, tunes weights, or edits predictions.
"""
from __future__ import annotations
import config as settings  # backend environment settings before numerical libraries
import argparse
import importlib.metadata
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from losses import psnr, ssim
from utils import (device_for, image_map, load_image, sha256_file, write_json,
                   environment_report)


def method_argument(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use --method NAME=PATH")
    name, path = text.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Method name and path cannot be empty")
    return name.strip(), Path(path.strip().strip('"'))


def scene_id(name):
    numbers = re.findall(r"\d+", str(name))
    if not numbers:
        raise ValueError(f"No numeric scene id: {name}")
    return int(numbers[-1])


def index_images(root, ocr=False):
    mapping = image_map(root)
    if not ocr:
        return mapping
    result, folder_ids = {}, {}
    for relative, path in mapping.items():
        parts = relative.split("/")
        if len(parts) < 2:
            raise ValueError(f"OCR images need scene subdirectories: {root} / {relative}")
        sid = scene_id(parts[0])
        if sid in folder_ids and folder_ids[sid] != parts[0]:
            raise ValueError(f"Multiple directories normalize to scene {sid} under {root}")
        folder_ids[sid] = parts[0]
        key = (sid, "/".join(parts[1:]))
        if key in result:
            raise ValueError(f"Duplicate OCR frame: {key}")
        result[key] = path
    return result


def load_labels(path):
    """Keep the source label order: sort words top-to-bottom then left-to-right."""
    data = pd.read_csv(path, keep_default_na=False)
    columns = list(data.columns)
    if "scene" not in columns or "text" not in columns:
        raise ValueError("labels CSV must have scene and text columns")
    start = max(columns.index("scene"), columns.index("text")) + 1
    coordinate_columns = columns[start:start + 4]
    if len(coordinate_columns) != 4:
        raise ValueError("Need four xyxy coordinate columns after scene/text")
    labels = defaultdict(list)
    for row in data.to_dict("records"):
        sid = int(row["scene"])
        box = tuple(float(row[k]) for k in coordinate_columns)
        if not all(math.isfinite(x) for x in box) or not (box[2] > box[0] and box[3] > box[1]):
            raise ValueError(f"Invalid label box: scene={sid}, box={box}")
        labels[sid].append({"text": str(row["text"]), "box": box})
    for sid, words in labels.items():
        words.sort(key=lambda w: (w["box"][1], w["box"][0]))
        for index, word in enumerate(words, 1):
            word["word_idx"] = index
    return dict(labels)


def crop_box(box, width, height, reference_width=512, reference_height=512, pad=8):
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, math.floor(x1 * width / reference_width) - pad))
    top = max(0, min(height - 1, math.floor(y1 * height / reference_height) - pad))
    right = max(left + 1, min(width, math.ceil(x2 * width / reference_width) + pad))
    bottom = max(top + 1, min(height, math.ceil(y2 * height / reference_height) + pad))
    return left, top, right, bottom


def normalize_text(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def lcs_length(a, b):
    previous = [0] * (len(b) + 1)
    for ca in a:
        current = [0]
        for index, cb in enumerate(b, 1):
            current.append(previous[index - 1] + 1 if ca == cb else max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + int(ca != cb)))
        previous = current
    return previous[-1]


def word_metrics(gt, pred):
    gt, pred = normalize_text(gt), normalize_text(pred)
    lcs, distance = lcs_length(pred, gt), edit_distance(pred, gt)
    return {
        "gt": gt, "pred": pred, "lcs": lcs, "edit_distance": distance,
        "nLCS": lcs / max(len(gt), 1),
        "NED_Sim": max(0.0, min(1.0, 1.0 - distance / max(len(gt), len(pred), 1))),
        "CER": distance / max(len(gt), 1), "recognized": int(bool(pred)),
        "exact": int(bool(gt) and gt == pred),
    }


def save_frame_summaries(rows, output, average):
    frame = pd.DataFrame(rows)
    metrics = [m for m in ("psnr", "ssim", "lpips", "niqe", "brisque") if m in frame]
    groups = frame.groupby(["method", "group"], as_index=False)[metrics].mean()
    groups["count"] = frame.groupby(["method", "group"])["key"].size().values
    source = groups if average == "group" else frame
    summaries = []
    for method, items in source.groupby("method", sort=False):
        selected = frame[frame["method"] == method]
        summaries.append({
            "method": method, "num_images": len(selected),
            "num_groups": int(selected["group"].nunique()), "averaging": average,
            **{m: float(items[m].mean()) for m in metrics},
        })
    frame.to_csv(output / "per_image_metrics.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(output / "group_metrics.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(summaries, output / "summary.json")
    print(summary.to_string(index=False))


def save_ocr_summaries(rows, output):
    frame = pd.DataFrame(rows)
    images = frame.groupby(["method", "scene_id", "frame_key"], as_index=False).agg(
        num_words=("word_idx", "count"), ROI_AD_LCS=("lcs", "sum"),
        nLCS=("nLCS", "mean"), NED_Sim=("NED_Sim", "mean"), WordAcc=("exact", "mean"),
        ROI_RecRate=("recognized", "mean"), CER=("CER", "mean"), MeanConfidence=("confidence", "mean"),
    )
    metric_names = ["ROI_AD_LCS", "nLCS", "NED_Sim", "WordAcc", "ROI_RecRate", "CER", "MeanConfidence"]
    scenes = images.groupby(["method", "scene_id"], as_index=False)[metric_names].mean()
    scenes["num_images"] = images.groupby(["method", "scene_id"])["frame_key"].size().values
    summaries = []
    for method, items in scenes.groupby("method", sort=False):
        summaries.append({
            "method": method, "num_scenes": len(items),
            "num_images": int((images["method"] == method).sum()),
            "num_word_instances": int((frame["method"] == method).sum()),
            **{m: float(items[m].mean()) for m in metric_names},
        })
    for name, table in (("per_word_metrics", frame), ("per_image_metrics", images), ("per_scene_metrics", scenes)):
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(summaries, output / "summary.json")
    print(summary.to_string(index=False))


def audit(args, methods, maps, reference, labels):
    expected = set(reference)
    details, checks = {}, []
    for name, root in methods.items():
        missing, extra = sorted(expected - set(maps[name])), sorted(set(maps[name]) - expected)
        details[name] = {"root": str(root), "count": len(maps[name]), "missing": missing[:10], "extra": extra[:10]}
        checks.append(not missing and not extra)
    if args.mode == "ocr":
        counts = defaultdict(int)
        for sid, _ in expected:
            counts[sid] += 1
        checks.append(set(counts) == set(labels))
        checks.append(args.expected_scenes == 0 or len(counts) == args.expected_scenes)
        checks.append(args.expected_frames == 0 or all(n == args.expected_frames for n in counts.values()))
        checks.append(args.expected_words == 0 or all(len(w) == args.expected_words for w in labels.values()))
    report = {"pass": all(checks), "mode": args.mode, "reference_count": len(expected), "methods": details,
              "options": vars(args), "environment": environment_report()}
    write_json(report, Path(args.output) / "protocol_audit.json")
    if not report["pass"]:
        raise ValueError("Exact-image/label audit failed. See protocol_audit.json. No samples were silently excluded.")
    return report


@torch.inference_mode()
def evaluate_images(args, maps, reference):
    device = device_for(args.device)
    metrics = {}
    if args.mode == "paired" and args.lpips:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("LPIPS requested: install lpips in your existing environment") from exc
        metrics["lpips"] = lpips.LPIPS(net="alex").to(device).eval()
    if args.mode == "noref":
        try:
            import pyiqa
        except ImportError as exc:
            raise RuntimeError("NIQE/BRISQUE requested: install pyiqa in your existing environment") from exc
        metrics = {name: pyiqa.create_metric(name, device=device, as_loss=False).eval()
                   for name in ("niqe", "brisque")}
    rows = []
    for key in tqdm(sorted(reference), desc=f"Test {args.mode}"):
        ref = load_image(reference[key], exif=args.mode == "noref").unsqueeze(0).to(device)
        for method, mapping in maps.items():
            prediction = load_image(mapping[key], exif=args.mode == "noref").unsqueeze(0).to(device)
            if prediction.shape != ref.shape:
                raise ValueError(f"Shape mismatch: {method} / {key}; no resizing is performed")
            p, g = prediction, ref
            border = args.crop_border
            if border:
                if min(p.shape[-2:]) <= 2 * border:
                    raise ValueError(f"crop_border too large for {key}")
                p, g = p[..., border:-border, border:-border], g[..., border:-border, border:-border]
            row = {"method": method, "key": key, "group": key.split("/")[0] if "/" in key else "__root__"}
            if args.mode == "paired":
                row.update(psnr=float(psnr(p, g)[0]), ssim=float(ssim(p, g)[0]))
                if "lpips" in metrics:
                    row["lpips"] = float(metrics["lpips"](p * 2 - 1, g * 2 - 1).mean())
            else:
                row.update({name: float(metric(p).detach().float().reshape(-1)[0]) for name, metric in metrics.items()})
            if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
                raise ValueError(f"Non-finite metric for {method} / {key}")
            rows.append(row)
    save_frame_summaries(rows, Path(args.output), args.average)


def load_existing_ocr(path, maps, labels):
    data = pd.read_csv(path, keep_default_na=False, dtype=str)
    required = {"method", "scene_id", "word_idx", "gt", "pred"}
    if not required <= set(data):
        raise ValueError(f"Missing OCR columns: {sorted(required - set(data))}")
    data = data[data["method"].isin(maps)]
    if "frame_key" not in data:
        if "image_rel" not in data:
            raise ValueError("OCR CSV needs frame_key or image_rel")
        keys = []
        for row in data.to_dict("records"):
            parts = row["image_rel"].replace("\\", "/").split("/")
            if len(parts) > 1 and scene_id(parts[0]) == int(row["scene_id"]):
                parts = parts[1:]
            keys.append(Path("/".join(parts)).with_suffix("").as_posix())
        data = data.copy()
        data["frame_key"] = keys
    expected = {(m, sid, frame, word["word_idx"])
                for m, mapping in maps.items() for sid, frame in mapping for word in labels[sid]}
    rows, seen = [], set()
    for record in data.to_dict("records"):
        sid, index = int(record["scene_id"]), int(record["word_idx"])
        key = (record["method"], sid, record["frame_key"].replace("\\", "/"), index)
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate OCR sample: {key}")
        label = labels[sid][index - 1]
        if normalize_text(label["text"]) != normalize_text(record["gt"]):
            raise ValueError(f"Ground truth mismatch for {key}")
        row = {"method": key[0], "scene_id": sid, "frame_key": key[2], "word_idx": index,
               "raw_pred": record.get("raw_pred", record["pred"]),
               "confidence": float(record.get("confidence") or 0.0),
               **word_metrics(record["gt"], record["pred"])}
        seen.add(key)
        rows.append(row)
    if seen != expected:
        raise ValueError(f"OCR CSV incomplete: missing {len(expected - seen)} words")
    return rows


@torch.inference_mode()
def recognize(args, maps, reference, labels):
    try:
        from doctr.models import recognition_predictor
    except ImportError as exc:
        raise RuntimeError("New OCR inference needs python-doctr; --ocr-csv reuses an existing per_word_metrics.csv") from exc
    predictor = recognition_predictor(arch="crnn_vgg16_bn", pretrained=True,
                                      symmetric_pad=True, batch_size=args.batch_size)
    predictor = predictor.to(device_for(args.device)).eval()
    rows, jobs = [], []

    def flush():
        if not jobs:
            return
        outputs = predictor([job[1] for job in jobs])
        if len(outputs) != len(jobs):
            raise RuntimeError("Recognizer output count mismatch")
        for (meta, _), item in zip(jobs, outputs):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                raw, confidence = str(item[0]), float(item[1])
            elif isinstance(item, dict):
                raw = str(item.get("value", item.get("text", item.get("label", ""))))
                confidence = float(item.get("confidence", item.get("conf", item.get("score", 0.0))))
            else:
                raise TypeError(f"Unsupported recognizer output type: {type(item)}")
            if not math.isfinite(confidence):
                raise ValueError("Non-finite OCR confidence")
            pred = normalize_text(raw) if confidence >= args.min_confidence else ""
            rows.append({**meta, "raw_pred": raw, "confidence": confidence, **word_metrics(meta["gt"], pred)})
        jobs.clear()

    for method, mapping in maps.items():
        for (sid, frame), path in tqdm(sorted(mapping.items()), desc=f"OCR {method}"):
            with Image.open(reference[(sid, frame)]) as ref:
                expected_size = ref.size
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB")
                if image.size != expected_size:
                    raise ValueError(f"OCR image size mismatch: {method} / {sid} / {frame}")
                for label in labels[sid]:
                    box = crop_box(label["box"], *image.size, args.coord_width, args.coord_height, args.pad)
                    # Native-resolution ROI; no detector, contrast changes or manual upscaling.
                    array = np.asarray(image.crop(box), dtype=np.uint8)
                    meta = {"method": method, "scene_id": sid, "frame_key": frame,
                            "filename": path.name, "word_idx": label["word_idx"], "gt": label["text"]}
                    jobs.append((meta, array))
                    if len(jobs) >= args.batch_size:
                        flush()
        flush()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["paired", "noref", "ocr"], required=True)
    parser.add_argument("--method", action="append", type=method_argument, required=True)
    parser.add_argument("--gt", "--gt_root", dest="gt", help="Paired GT directory, exact relative-key mirror")
    parser.add_argument("--input", dest="reference", help="Optional noref reference-input root")
    parser.add_argument("--output", "--output_dir", dest="output", default="results/test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--average", choices=["image", "group"], default="group")
    parser.add_argument("--crop-border", type=int, default=0)
    parser.add_argument("--lpips", action="store_true", help="Paired: also compute LPIPS-Alex")
    parser.add_argument("--audit-only", action="store_true", help="Check file keys/labels without metric engines")
    parser.add_argument("--labels", help="OCR labels_final.csv")
    parser.add_argument("--ocr-csv", help="OCR: aggregate existing per-word outputs instead of rerunning CRNN")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--coord-width", type=float, default=512)
    parser.add_argument("--coord-height", type=float, default=512)
    parser.add_argument("--pad", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--expected-scenes", type=int, default=100)
    parser.add_argument("--expected-frames", type=int, default=100)
    parser.add_argument("--expected-words", type=int, default=5)
    args = parser.parse_args()
    if args.crop_border < 0 or args.pad < 0 or min(args.batch_size, args.coord_width, args.coord_height) <= 0:
        parser.error("Invalid crop/batch/coordinate settings")
    methods = dict(args.method)
    if len(methods) != len(args.method):
        parser.error("Duplicate --method names are not allowed")
    if args.mode == "paired" and not args.gt:
        parser.error("Paired mode requires --gt")
    if args.mode == "ocr" and (not args.labels or "Input" not in methods):
        parser.error("OCR mode requires --labels and --method Input=PATH")
    if args.mode == "ocr" and args.crop_border:
        parser.error("OCR uses annotated ROI cropping, not image-border cropping")
    output = Path(args.output).resolve()
    for root in list(methods.values()) + [Path(p) for p in (args.gt, args.reference) if p]:
        root = root.resolve()
        if root == output or root in output.parents or output in root.parents:
            parser.error("Evaluation output must be separate from image/GT directories")
    if args.ocr_csv and Path(args.ocr_csv).resolve().parent == output:
        parser.error("Choose a different output directory to preserve the source OCR CSV")
    output.mkdir(parents=True, exist_ok=True)
    maps = {m: index_images(p, args.mode == "ocr") for m, p in methods.items()}
    if args.mode == "paired":
        reference = image_map(args.gt)
    elif args.mode == "ocr":
        reference = maps["Input"]
    else:
        reference = image_map(args.reference) if args.reference else maps.get("Input", next(iter(maps.values())))
    labels = load_labels(args.labels) if args.mode == "ocr" else None
    report = audit(args, methods, maps, reference, labels)
    if args.labels:
        report["labels_sha256"] = sha256_file(args.labels)
    if args.ocr_csv:
        report["ocr_csv_sha256"] = sha256_file(args.ocr_csv)
        report["ocr_source_note"] = "Existing CSV reused; model/preprocessing provenance remains with the original OCR run."
    report["packages"] = {}
    for name in ("Pillow", "pandas", "lpips", "pyiqa", "python-doctr"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    write_json(report, output / "protocol_audit.json")
    if args.audit_only:
        print("File-key/label audit PASSED (does not validate pixel decoding or metric engines)")
        return
    if args.mode == "ocr":
        rows = load_existing_ocr(args.ocr_csv, maps, labels) if args.ocr_csv else recognize(args, maps, reference, labels)
        save_ocr_summaries(rows, output)
    else:
        evaluate_images(args, maps, reference)
    report["evaluation_completed"] = True
    write_json(report, output / "protocol_audit.json")


if __name__ == "__main__":
    main()
