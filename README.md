# Road Scene Risk Analysis

A Python video analysis project that detects and tracks road users, segments the drivable area, estimates scene depth, and assigns each detection a simple proximity risk level. The pipeline saves an annotated video and a frame-by-frame JSON Lines log.

## Project structure

Run the script from this directory so its relative model paths resolve correctly.

```text
Road_Scene_Know-main/
├── README.md                       # this guide
├── pipeline.py                     # video inference and risk scoring
├── best-2.pt                       # default YOLO detector weights
├── input_video.mp4                  # sample input video
├── out.mp4                          # sample annotated output
├── out.jsonl                        # sample per-frame log
└── segformer_idd/                   # local semantic segmentation model
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

The sample `out.mp4` and `out.jsonl` are already-produced examples; running the default command overwrites them. Choose a different `--out` path to keep those examples.

## What the pipeline does

[`pipeline.py`](pipeline.py) combines three model stages:

1. **Object detection and tracking:** Ultralytics YOLO detects objects using `best-2.pt`, and ByteTrack assigns IDs across video frames.
2. **Drivable-area segmentation:** a local SegFormer model predicts seven semantic classes. Segmentation class `0` is treated as drivable.
3. **Depth and risk estimation:** Depth Anything V2 Metric Outdoor Small estimates depth. For each detection, the script samples the lower central region of its box, estimates closing speed from recent depth estimates, checks whether the object is on the road and in the ego path, and assigns a risk level.

Detection and tracking run on every frame. Segmentation runs every three frames and depth every two frames by default; the most recent prediction is reused in between. The cadence can be changed through command-line options.

## Requirements

- Python 3.9 or later is recommended.
- Internet access is needed on the first run to fetch the depth model configured in `pipeline.py`. YOLO and SegFormer weights are included in this project directory.
- OpenCV must support the input video's codec and MP4 output.
- The code automatically selects Apple MPS, NVIDIA CUDA, or CPU. A supported GPU can speed up inference; CPU processing may be slow.

## Installation

Open a terminal in this directory, then create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python packages imported by the script:

```bash
python -m pip install --upgrade pip
python -m pip install numpy opencv-python torch transformers ultralytics
```

For GPU inference, install a PyTorch build that matches your CUDA or Apple Silicon setup using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Depending on your system, the command above may install a CPU-only build.

## Run the sample

From this directory, run:

```powershell
python pipeline.py --src input_video.mp4 --out results/annotated.mp4
```

Create the `results` directory before running; the script does not create parent directories. This produces:

- `results/annotated.mp4` — annotated video.
- `results/annotated.jsonl` — per-frame analysis log.

Use paths to another video or detector checkpoint as needed:

```powershell
python pipeline.py --src path/to/video.mp4 --out results/video.mp4 --weights path/to/detector.pt
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--src` | Required | Input video path. |
| `--out` | `out.mp4` | Annotated video output path. The matching `.jsonl` filename is derived from it. |
| `--weights` | `best-2.pt` | YOLO detector checkpoint path. |
| `--imgsz` | `640` | YOLO inference image size. |
| `--seg_every` | `3` | Run segmentation once every N frames. |
| `--depth_every` | `2` | Run depth estimation once every N frames. |
| `--conf` | `0.25` | YOLO detection confidence threshold. |

For example, run segmentation and depth on every frame with `--seg_every 1 --depth_every 1`. This increases processing time.

## Understanding the outputs

### Annotated video

Drivable-area pixels are overlaid in translucent green. Each detection gets a bounding box and a label containing its YOLO class name, track ID, estimated distance in meters, and risk level. Box colors are red for `HIGH`, orange for `MED`, and green for `LOW`.

Risk levels use fixed rules in `pipeline.py`:

- **HIGH:** the detection is on the road and in the ego path, and either estimated distance is under 8 m or time-to-collision (TTC) is under 2 s.
- **MED:** the detection is on the road, and either distance is under 15 m or TTC is under 4 s.
- **LOW:** neither condition is met.

Closing speed is calculated from a linear fit to recent distance estimates. TTC is calculated only when estimated closing speed is above 0.3 m/s; otherwise it is infinite internally and recorded as `null` in JSONL.

### JSON Lines log

Each line in the `.jsonl` file is a JSON object for one zero-based video frame. It contains a `frame` index and an `objects` array. Object entries have these fields:

| Field | Meaning |
| --- | --- |
| `id` | ByteTrack ID, or `-1` when no ID is available. |
| `cls` | Object class name supplied by the YOLO checkpoint. |
| `dist_m` | Median depth estimate from the lower-central portion of the detection box, in meters according to the selected metric depth model. |
| `closing_mps` | Estimated positive closing speed in meters per second. |
| `ttc_s` | Estimated time-to-collision in seconds, or `null` when closing speed is at or below the threshold. |
| `on_road` | Whether more than half of the sampled strip below the box is classified as drivable. |
| `in_path` | Whether the box center is inside the middle 60 percent of the drivable span. |
| `risk` | `HIGH`, `MED`, or `LOW`. |

Example record shape:

```json
{"frame": 0, "objects": [{"id": 1, "cls": "car", "dist_m": 12.4, "closing_mps": 0.0, "ttc_s": null, "on_road": true, "in_path": true, "risk": "MED"}]}
```

## Assumptions and limitations

- SegFormer config labels are generic (`LABEL_0` through `LABEL_6`). The script assumes class `0` is drivable and resizes frames to 576 by 768 for segmentation. Confirm that class ordering and preprocessing match the model's training setup.
- Distance and closing speed are model estimates, not calibrated sensor readings. Risk scoring does not account for camera calibration, vehicle motion, road geometry, or prediction uncertainty.
- Tracking and distance history exist only during one processing run.
- This project is for experimentation and offline video analysis. It is not a certified driver assistance system and should not be used to make real-world driving decisions.

## Troubleshooting

- **Depth model fails to download or load:** check internet access and the model ID configured as `DEPTH_ID` in `pipeline.py`; model files are cached by Hugging Face after download.
- **YOLO checkpoint not found:** run the script from this directory or provide the correct path using `--weights`.
- **SegFormer model not found:** run the script from this directory; the script loads the relative `segformer_idd/` folder.
- **CUDA out-of-memory:** reduce `--imgsz`, use a smaller input video, or use CPU inference.
- **Video cannot be opened or saved:** verify the input path and the codecs supported by your OpenCV installation.
- **Output path error:** create the output's parent folder before running.

## License and model attribution

No project license or model-training attribution is included in the files inspected. Before redistributing the project or using it commercially, review the terms for `best-2.pt`, the local SegFormer checkpoint, the configured Hugging Face depth model, and their training data.
