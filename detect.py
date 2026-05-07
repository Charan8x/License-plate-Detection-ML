import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from collections import defaultdict, Counter
import os
import sys
import time


VIDEO_PATH      = "input.mp4"
OUTPUT_PATH     = "output.mp4"
CONF_THRESHOLD  = 0.45
IOU_THRESHOLD   = 0.5
OCR_EVERY_N     = 5
VOTE_AFTER      = 4
MIN_TEXT_LEN    = 3
BOX_COLOR       = (0, 220, 0)
TEXT_COLOR      = (0, 220, 0)
TEXT_BG_COLOR   = (0, 0, 0)
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.75
FONT_THICKNESS  = 2



def download_model():
    model_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_file = os.path.join(model_dir, "license-plate-finetune-v1x.pt")
    if os.path.exists(model_file):
        print(f"[✓] Model found: {model_file}")
        return model_file
    print("[↓] Downloading YOLOv11x from Hugging Face ...")
    os.makedirs(model_dir, exist_ok=True)
    model_file = hf_hub_download(
        repo_id   = "morsetechlab/yolov11-license-plate-detection",
        filename  = "license-plate-finetune-v1x.pt",
        local_dir = model_dir,
    )
    print(f"[✓] Model saved: {model_file}")
    return model_file


def load_models():
    model_path = download_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[✓] Device: {device.upper()}")
    if device == "cuda":
        print(f"    GPU  : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    VRAM : {vram:.1f} GB")

    print("[↓] Loading YOLO model ...")
    yolo = YOLO(model_path)

    print("[↓] Loading EasyOCR ...")
    reader = easyocr.Reader(
        ["en"],
        gpu     = torch.cuda.is_available(),
        verbose = False,
    )

    print("[✓] All models ready\n")
    return yolo, reader, device


def get_variants(crop_bgr):
    """
    Return multiple preprocessed versions of the crop.
    EasyOCR will run on all of them — best result wins.
    """
    h, w = crop_bgr.shape[:2]
    variants = []

    # Always upscale to at least 200px wide
    scale = max(1.0, 200 / w)
    big   = cv2.resize(crop_bgr,
                       (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_CUBIC)
    variants.append(big)

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    # V1: CLAHE equalisation (handles dark/light plates)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    v_clahe = clahe.apply(gray)
    variants.append(cv2.cvtColor(v_clahe, cv2.COLOR_GRAY2BGR))

    # V2: Sharpened
    kernel   = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    v_sharp  = cv2.filter2D(gray, -1, kernel)
    variants.append(cv2.cvtColor(v_sharp, cv2.COLOR_GRAY2BGR))

    # V3: Otsu threshold (clean black/white)
    _, v_otsu = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(v_otsu, cv2.COLOR_GRAY2BGR))

    # V4: Inverted Otsu (for dark plates with light text)
    variants.append(cv2.cvtColor(255 - v_otsu, cv2.COLOR_GRAY2BGR))

    return variants


def clean(text):
    return "".join(c for c in text.upper() if c.isalnum() or c == " ").strip()


def run_ocr(reader, crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    best = ""
    for variant in get_variants(crop_bgr):
        try:
            results = reader.readtext(
                variant,
                detail          = 1,
                paragraph       = False,
                allowlist       = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
                text_threshold  = 0.6,
                low_text        = 0.3,
                link_threshold  = 0.4,
            )
        except Exception:
            continue

        # Only keep high-confidence detections
        parts = [res[1] for res in results if res[2] > 0.5]
        text  = clean(" ".join(parts))

        if len(text) > len(best):
            best = text

    return best


def best_voted_text(history):
    valid = [t for t in history if len(t) >= MIN_TEXT_LEN]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]


def draw_label(frame, text, x1, y1):
    label        = f" {text} " if text else " ? "
    (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    ly1 = max(y1 - th - bl - 8, 0)
    ly2 = ly1 + th + bl + 8
    lx2 = min(x1 + tw + 4, frame.shape[1])
    cv2.rectangle(frame, (x1, ly1), (lx2, ly2), TEXT_BG_COLOR, -1)
    cv2.rectangle(frame, (x1, ly1), (lx2, ly2), BOX_COLOR, 1)
    cv2.putText(frame, label, (x1 + 2, ly2 - bl - 2),
                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


def make_cache():
    return {
        "text"     : "",
        "history"  : [],
        "locked"   : False,
        "last_ocr" : -999,
        "ocr_count": 0,
    }


def process_video(video_path, output_path, yolo, reader, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[✗] Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[✓] Video  : {width}x{height} @ {fps:.1f} fps — {total_frames} frames")
    print(f"[✓] Output : {os.path.abspath(output_path)}\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cache      = defaultdict(make_cache)
    frame_idx  = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = yolo.track(
            frame,
            conf    = CONF_THRESHOLD,
            iou     = IOU_THRESHOLD,
            device  = device,
            persist = True,
            verbose = False,
            tracker = "bytetrack.yaml",
        )

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf_score      = float(box.conf[0])
                track_id        = int(box.id[0]) if box.id is not None else -1
                entry           = cache[track_id]

                since_last = frame_idx - entry["last_ocr"]
                if not entry["locked"] and since_last >= OCR_EVERY_N:
                    pad  = 4
                    crop = frame[
                        max(0, y1 - pad) : min(height, y2 + pad),
                        max(0, x1 - pad) : min(width,  x2 + pad),
                    ]

                    new_text           = run_ocr(reader, crop)
                    entry["last_ocr"]  = frame_idx
                    entry["ocr_count"] += 1

                    if len(new_text) >= MIN_TEXT_LEN:
                        entry["history"].append(new_text)

                    voted = best_voted_text(entry["history"])
                    if voted:
                        entry["text"] = voted

                    if entry["ocr_count"] >= VOTE_AFTER and entry["text"]:
                        entry["locked"] = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                draw_label(frame, entry["text"], x1, y1)
                cv2.putText(frame, f"{conf_score:.2f}",
                            (x2 - 48, y2 + 18),
                            FONT, 0.5, BOX_COLOR, 1, cv2.LINE_AA)

        elapsed  = time.time() - start_time
        live_fps = frame_idx / elapsed if elapsed > 0 else 0
        cv2.putText(frame,
                    f"Frame {frame_idx}/{total_frames}  |  {live_fps:.1f} fps",
                    (10, 28), FONT, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(frame)

        pct = frame_idx / total_frames * 100
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:.1f}%  ({frame_idx}/{total_frames})")
        sys.stdout.flush()

    cap.release()
    out.release()
    total_time = time.time() - start_time
    print(f"\n\n[✓] Done in {total_time:.1f}s")
    print(f"[✓] Saved  → {os.path.abspath(output_path)}")


def main():
    print("=" * 55)
    print("  License Plate Detection  |  YOLOv11x + EasyOCR")
    print("=" * 55 + "\n")

    if not os.path.exists(VIDEO_PATH):
        print(f"[✗] Video not found: '{VIDEO_PATH}'")
        sys.exit(1)

    yolo, reader, device = load_models()
    process_video(VIDEO_PATH, OUTPUT_PATH, yolo, reader, device)


if __name__ == "__main__":
    main()