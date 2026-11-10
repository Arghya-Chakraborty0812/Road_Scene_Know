import argparse, json, collections
import cv2, numpy as np, torch
import torch.nn.functional as F
from ultralytics import YOLO
from transformers import (SegformerImageProcessor, SegformerForSemanticSegmentation,
                          AutoImageProcessor, AutoModelForDepthEstimation)

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
SEG_ID = "segformer_idd"                                               # local folder
DEPTH_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"  # verify id on HF
DRIVABLE = 0   # 0 drivable, 1 non-drivable, 2 living, 3 vehicle, 4 roadside, 5 far, 6 sky


class Segmenter:
    def __init__(self, size=(576, 768)):                # (h, w): same scale as training
        self.size = size
        self.proc = SegformerImageProcessor.from_pretrained(SEG_ID)
        self.model = SegformerForSemanticSegmentation.from_pretrained(SEG_ID).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.size[1], self.size[0])), cv2.COLOR_BGR2RGB)
        inp = self.proc(images=rgb, return_tensors="pt", do_resize=False).to(DEVICE)
        logits = self.model(**inp).logits
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


class Depth:
    def __init__(self):
        self.proc = AutoImageProcessor.from_pretrained(DEPTH_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.proc(images=rgb, return_tensors="pt").to(DEVICE)
        d = self.model(**inp).predicted_depth                  # meters (metric model)
        d = F.interpolate(d[None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        return d.cpu().numpy().astype(np.float32)


def object_distance(depth, box):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    roi = depth[(y1 + y2) // 2: y2, x1 + int(0.2 * bw): x2 - int(0.2 * bw)]   # lower-central part
    return float(np.median(roi)) if roi.size else float("nan")


def on_road(seg, box):
    x1, y1, x2, y2 = box
    H = seg.shape[0]
    strip = seg[min(y2, H - 1): min(y2 + 12, H), x1 + (x2 - x1) // 4: x2 - (x2 - x1) // 4]
    return bool(strip.size) and float((strip == DRIVABLE).mean()) > 0.5


def in_ego_path(seg, box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    row = seg[min(y2 + 10, seg.shape[0] - 1)]
    xs = np.where(row == DRIVABLE)[0]
    if len(xs) == 0:
        return False
    lo, hi = np.percentile(xs, [20, 80])                 # middle of the drivable span
    return bool(lo <= cx <= hi)


def closing_speed(hist):
    if len(hist) < 5:
        return 0.0
    t, d = zip(*hist)
    return -float(np.polyfit(t, d, 1)[0])                # m/s, positive = getting closer


def risk_level(dist, closing, road, in_path):
    ttc = dist / closing if closing > 0.3 else float("inf")
    if road and in_path and (dist < 8 or ttc < 2):
        return "HIGH", ttc
    if road and (dist < 15 or ttc < 4):
        return "MED", ttc
    return "LOW", ttc


COLORS = {"HIGH": (0, 0, 255), "MED": (0, 165, 255), "LOW": (0, 200, 0)}


def run(src, out, weights, imgsz, seg_every, depth_every, conf):
    det = YOLO(weights)
    seg_model, depth_model = Segmenter(), Depth()

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    log = open(out.rsplit(".", 1)[0] + ".jsonl", "w")

    hist = collections.defaultdict(lambda: collections.deque(maxlen=15))
    smooth = {}
    seg = depth = None
    i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if seg is None or i % seg_every == 0:
            seg = seg_model(frame)
        if depth is None or i % depth_every == 0:
            depth = depth_model(frame)

        res = det.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=imgsz,
                        conf=conf, device=DEVICE, verbose=False)[0]

        overlay = frame.copy()
        overlay[seg == DRIVABLE] = (0, 180, 0)           # drivable area in green
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        objs = []
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, W - 1), min(y2, H - 1)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            tid = int(b.id) if b.id is not None else -1
            name = det.names[int(b.cls)]

            d = object_distance(depth, (x1, y1, x2, y2))
            if np.isnan(d):
                continue
            if tid >= 0:
                d = 0.6 * smooth.get(tid, d) + 0.4 * d    # EMA smoothing
                smooth[tid] = d
                hist[tid].append((i / fps, d))
            closing = closing_speed(hist[tid]) if tid >= 0 else 0.0

            road = on_road(seg, (x1, y1, x2, y2))
            in_path = in_ego_path(seg, (x1, y1, x2, y2))
            level, ttc = risk_level(d, closing, road, in_path)

            color = COLORS[level]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}#{tid} {d:.1f}m {level}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            objs.append(dict(id=tid, cls=name, dist_m=round(d, 2), closing_mps=round(closing, 2),
                             ttc_s=None if ttc == float("inf") else round(ttc, 2),
                             on_road=bool(road), in_path=in_path, risk=level))

        log.write(json.dumps(dict(frame=i, objects=objs)) + "\n")
        writer.write(frame)
        i += 1
        if i % 20 == 0:
            print(f"frame {i}", flush=True)

    cap.release(); writer.release(); log.close()
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--weights", default="best-2.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seg_every", type=int, default=3)
    ap.add_argument("--depth_every", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    run(a.src, a.out, a.weights, a.imgsz, a.seg_every, a.depth_every, a.conf)