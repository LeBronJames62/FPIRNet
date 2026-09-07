# FPIR-Net

Run all commands from the project root directory. Install PyTorch before running the code, then install the remaining core dependencies:

```bash
python -m pip install -r requirements.txt
```

Training, inference, and testing use `main_train.py`, `inference.py`, and `main_test.py`, respectively.

## 1. Training

### Data Preparation

Place clean images in the training and validation directories. The training pipeline generates turbulence degradation on the fly, so precomputed pairs of degraded and clean images are not required.

```text
data/
└── clean/
    ├── train/
    │   ├── image_001.png
    │   └── ...
    └── val/
        ├── image_001.png
        └── ...
```

### Start Training

```bash
python main_train.py --train-root data/clean/train --val-root data/clean/val --output ckpt/train
```

Training settings are defined in `config.py`. The default configuration uses 4 recurrent stages, 192×192 crops, a batch size of 1, gradient accumulation over 4 batches, up to 5,000 optimizer updates, AdamW, and an exponential moving average (EMA) of the model weights. Common settings can be overridden from the command line:

```bash
python main_train.py --train-root data/clean/train --val-root data/clean/val --output ckpt/train_custom --max-steps 5000 --epochs 6 --batch-size 1 --accumulation 4 --patch-size 192 --workers 2 --precision bf16
```

### Resume Training

```bash
python main_train.py --train-root data/clean/train --val-root data/clean/val --output ckpt/train --resume ckpt/train/latest.pth
```

When resuming, keep the data and training settings in `config.py` and the command-line arguments consistent with the original run. Use a separate output directory for each new training run to avoid overwriting existing checkpoints.

### Training Outputs

```text
ckpt/train/
├── best.pth
├── latest.pth
├── best_metrics.json
├── validation.json
├── config.json
├── environment.json
└── train.log
```

`best.pth` is selected by validation PSNR and is used for image restoration inference. `latest.pth` is used to resume training.

## 2. Inference

`inference.py` reads degraded images and model weights and saves restored images. It does not read ground-truth (GT) images or compute evaluation metrics.

### Folder Inference

```bash
python inference.py --checkpoint ckpt/train/best.pth --input data/test --output results/restored
```

Alternatively, place an existing checkpoint at `ckpt/best.pth` to use the default checkpoint path:

```bash
python inference.py --input data/test --output results/restored
```

The input directory may contain subdirectories. Restored images retain the original resolution and relative directory structure and are saved as PNG files. Inference settings and completion status are recorded in `inference.json` in the output directory. The input and output directories must be separate: they must not be identical or nested within one another.

The default settings use CUDA, BF16 precision, 4 recurrent stages, a tile size of 384 pixels, and an overlap of 64 pixels. EMA weights are used by default when they are available in a training checkpoint. To specify the inference settings explicitly:

```bash
python inference.py --checkpoint ckpt/train/best.pth --input data/test --output results/restored --iterations 4 --tile 384 --overlap 64 --device cuda --precision bf16
```

For CPU inference:

```bash
python inference.py --checkpoint ckpt/train/best.pth --input data/test --output results/restored_cpu --device cpu --precision fp32
```

### Resume Incomplete Inference

Add `--resume` when the input images, checkpoint, and inference settings are unchanged:

```bash
python inference.py --checkpoint ckpt/train/best.pth --input data/test --output results/restored --resume
```

The script checks existing outputs and continues processing unfinished images. Use a new output directory when changing the checkpoint or inference settings.

If a checkpoint contains additional objects that prevent loading with the default safe loader, add `--trusted-checkpoint` only for files you created and trust. Do not use this option for files from unknown sources.

## 3. Testing

`main_test.py` evaluates saved images without invoking FPIR-Net for restoration. Use `--mode` to select paired, no-reference, or OCR evaluation. Compare multiple methods by repeating `--method "NAME=DIRECTORY"`.

### 3.1 Paired Evaluation: PSNR, SSIM, and LPIPS

GT images and the outputs of every method must have matching relative paths, excluding file extensions, and matching image dimensions. File extensions may differ, but duplicate extension-independent relative paths within a dataset root are not allowed.

```text
data/test_gt/scene_001/frame_001.png
results/restored/scene_001/frame_001.png
```

To compute PSNR and SSIM:

```bash
python main_test.py --mode paired --gt data/test_gt --method "Ours=results/restored" --output results/eval_paired --average image
```

To compute LPIPS as well, install its dependency and add `--lpips`:

```bash
python -m pip install lpips
```

```bash
python main_test.py --mode paired --gt data/test_gt --method "Ours=results/restored" --output results/eval_paired_lpips --average image --lpips
```

Example of group-balanced evaluation with multiple methods:

```bash
python main_test.py --mode paired --gt data/CLEAR/gt --method "Input=data/CLEAR/turb" --method "Ours=results/CLEAR" --method "BaryIR=results/BaryIR_CLEAR" --output results/eval_CLEAR --average group --lpips
```

`--average image` averages over all images. `--average group` first averages within each first-level subdirectory and then averages the group means with equal weights. By default, evaluation uses RGB images without border cropping. Use `--crop-border` to apply the same border crop to all methods.

### 3.2 No-Reference Evaluation: NIQE and BRISQUE

Install the dependency:

```bash
python -m pip install pyiqa
```

Example:

```bash
python main_test.py --mode noref --method "Input=data/OTIS" --method "Ours=results/OTIS" --output results/eval_OTIS --average group
```

All methods must contain the same image set with matching image dimensions. This mode does not require clean GT images. When `Input` is provided, its file set is used to check correspondence across methods.

### 3.3 OCR Evaluation

Organize input and restored images by scene. The label file must contain `scene` and `text`, immediately followed by four coordinate columns in the order `x1, y1, x2, y2`.

```text
data/text/
├── labels_final.csv
└── Input/
    ├── scene_001/
    │   ├── frame_001.png
    │   └── ...
    └── ...

results/Text/
├── scene_001/
│   ├── frame_001.png
│   └── ...
└── ...
```

First, generate the restored images:

```bash
python inference.py --checkpoint ckpt/train/best.pth --input data/text/Input --output results/Text
```

Install the OCR dependency, then use the frozen `crnn_vgg16_bn` recognizer on the fixed word regions:

```bash
python -m pip install python-doctr
```

```bash
python main_test.py --mode ocr --labels data/text/labels_final.csv --method "Input=data/text/Input" --method "Ours=results/Text" --output results/eval_Text --batch-size 128
```

By default, label coordinates use a 512×512 reference size, word boxes are expanded with `pad=8`, and no additional text detection or manual preprocessing is performed. The default checks require 100 scenes, 100 frames per scene, and 5 annotated words per frame. For datasets of other sizes, set `--expected-scenes`, `--expected-frames`, and `--expected-words` accordingly.

If a per-word recognition CSV is already available, recompute the summaries without running the recognizer:

```bash
python main_test.py --mode ocr --labels data/text/labels_final.csv --method "Input=data/text/Input" --method "Ours=results/Text" --ocr-csv results/ocr_cache/per_word_metrics.csv --output results/eval_Text_cached
```

The CSV must contain `method`, `scene_id`, `word_idx`, `gt`, and `pred`, together with either `frame_key` or `image_rel`. Method names must match those passed through `--method`, and the records must cover all images and annotated words in the current evaluation. The output directory must differ from the directory containing the source CSV.

OCR outputs include nLCS, NED-Sim, WordAcc, ROI_RecRate, and CER, with aggregation performed successively at the word, frame, and scene levels. `WordAcc` is the exact word-match rate. `ROI_RecRate` is the proportion of nonempty predictions after thresholding.

### Test Outputs

Paired and no-reference evaluations save `per_image_metrics.csv`, `group_metrics.csv`, `summary_metrics.csv`, and `summary.json`. OCR evaluation saves `per_word_metrics.csv`, `per_image_metrics.csv`, `per_scene_metrics.csv`, `summary_metrics.csv`, and `summary.json`. All modes save `protocol_audit.json`.

The evaluation output directory must be separate from the image and GT directories. The script checks file correspondence across methods and does not automatically skip missing samples or resize images. Add `--audit-only` to check file keys and label correspondence before evaluation. This option does not compute image metrics or run OCR, and it does not validate image decoding or metric weights.

LPIPS, NIQE/BRISQUE, and new OCR inference require the pretrained resources used by their respective libraries. Ensure that these resources can be downloaded at runtime or are already available in the local cache.
