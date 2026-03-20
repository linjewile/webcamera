"""
crowd_tags.py
─────────────
Reads crowd video footage, detects and tracks every person using YOLOv8,
and overlays a persistent life-status tag above each person for as long
as they stay in the frame.

Re-identification strategy (three layers):
  1. BoT-SORT tracker  — uses Kalman prediction + appearance ReID internally
  2. Ghost buffer      — remembers lost tracks for GHOST_FRAMES frames;
                         new detections are matched by appearance histogram
                         + last-known position, so a person walking back into
                         frame gets their original tag back
  3. Tag registry      — tag/colour is keyed on track ID so it never changes

Export: 3K wide (3072 px), 24 fps, H.264 via ffmpeg if available,
        falling back to mp4v otherwise.

Usage:
    python crowd_tags.py                      # all videos in ./webcamera/
    python crowd_tags.py myvideo.mp4          # single file
    python crowd_tags.py a.mp4 b.mov c.MP4   # multiple files

    --conf   0.30   detection confidence (lower = more detections in crowds)
    --model  yolov8s.pt   use a bigger model for better accuracy

Requirements:
    pip install ultralytics opencv-python pillow numpy
"""

import cv2
import numpy as np
import random
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
#  Export settings
# ─────────────────────────────────────────────────────────────────────────────
OUT_WIDTH  = 3072   # 3K wide
OUT_FPS    = 24

# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID: ghost buffer settings
# ─────────────────────────────────────────────────────────────────────────────
GHOST_FRAMES      = 72   # frames to remember a lost track (3 s at 24 fps)
REID_APP_WEIGHT   = 0.65  # weight of appearance similarity in matching score
REID_IOU_WEIGHT   = 0.35  # weight of position overlap in matching score
REID_MIN_SCORE    = 0.42  # minimum score to accept a re-ID match

# ─────────────────────────────────────────────────────────────────────────────
#  Life-status tags
# ─────────────────────────────────────────────────────────────────────────────
TAGS = [
    # Work / money
    "just got fired",
    "got a promotion today",
    "secretly a millionaire",
    "owes $47k in student loans",
    "just asked for a raise",
    "hasn't filed taxes in 3 years",
    "just quit their job",
    "waiting on a job offer",
    "starting a startup",
    "about to be acquired",
    # Relationships
    "first date tonight",
    "just got dumped",
    "getting married next week",
    "texting their ex",
    "in love with their best friend",
    "just got cheated on",
    "hasn't texted back in 3 days",
    "about to propose",
    "just had a baby",
    "going through a divorce",
    # Life moments
    "hasn't slept in 2 days",
    "just found out they're pregnant",
    "just got their heart broken",
    "running late",
    "forgot someone's birthday",
    "just moved to a new city",
    "going through it",
    "best day of their life",
    "worst week of their life",
    "just got diagnosed",
    # Fun / absurd
    "thinks nobody knows",
    "main character",
    "side character",
    "peak of their life",
    "already peaked",
    "definitely not a robot",
    "just saw something they can't unsee",
    "on their 4th coffee",
    "pretending to be fine",
    "absolutely thriving",
    "mid-life crisis incoming",
    "just manifested something",
    "not who you think they are",
    "has a secret",
    "just changed their mind",
]

PALETTES = [
    ((15,  15,  15,  210), (255, 255, 255)),   # black / white
    ((245, 245, 245, 215), (20,  20,  20 )),   # white / black
    ((18,  22,  45,  215), (160, 190, 255)),   # dark-blue / soft-blue
    ((42,  10,  10,  215), (255, 170, 170)),   # dark-red  / soft-red
    ((8,   38,  18,  215), (130, 255, 170)),   # dark-green / soft-green
    ((38,  26,  4,   215), (255, 210, 120)),   # dark-amber / soft-amber
]


# ─────────────────────────────────────────────────────────────────────────────
#  Track state + ghost buffer  (reset per video)
# ─────────────────────────────────────────────────────────────────────────────
class TrackState:
    __slots__ = ("tag", "palette", "last_box", "last_frame", "appearance")

    def __init__(self):
        self.tag        = random.choice(TAGS)
        self.palette    = random.choice(PALETTES)
        self.last_box   = None   # (x1,y1,x2,y2) in detection-resolution coords
        self.last_frame = 0
        self.appearance = None   # HSV histogram ndarray


_registry: dict[int, TrackState] = {}  # active tracks
_ghosts:   dict[int, TrackState] = {}  # recently-lost tracks


def _reset_state():
    _registry.clear()
    _ghosts.clear()


def _get_state(track_id: int) -> TrackState:
    if track_id not in _registry:
        _registry[track_id] = TrackState()
    return _registry[track_id]


# ─────────────────────────────────────────────────────────────────────────────
#  Appearance + position helpers
# ─────────────────────────────────────────────────────────────────────────────
def _appearance(frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray | None:
    """16-bin HSV hue + 8-bin saturation histogram of the person crop."""
    crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _app_sim(h1, h2) -> float:
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(
        h1.reshape(-1, 1).astype(np.float32),
        h2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL,
    ))


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _expire_ghosts(current_frame: int):
    expired = [gid for gid, gs in _ghosts.items()
               if current_frame - gs.last_frame > GHOST_FRAMES]
    for gid in expired:
        del _ghosts[gid]


def _match_ghost(box, appearance, current_frame: int) -> int | None:
    """Return the ghost track_id that best matches this new detection, or None."""
    _expire_ghosts(current_frame)
    best_id, best_score = None, REID_MIN_SCORE
    for gid, gs in _ghosts.items():
        if gs.last_box is None:
            continue
        score = (REID_APP_WEIGHT * _app_sim(appearance, gs.appearance)
                 + REID_IOU_WEIGHT * _iou(box, gs.last_box))
        if score > best_score:
            best_score, best_id = score, gid
    return best_id


def _on_track_seen(track_id: int, box, frame: np.ndarray, frame_idx: int):
    """Called every frame a track is visible — updates state and resolves ghost."""
    # If this is a brand-new ID, check the ghost buffer first
    if track_id not in _registry:
        x1, y1, x2, y2 = box
        app   = _appearance(frame, x1, y1, x2, y2)
        ghost = _match_ghost(box, app, frame_idx)
        if ghost is not None:
            # Resurrect old state under new tracker ID
            _registry[track_id] = _ghosts.pop(ghost)
        else:
            _registry[track_id] = TrackState()

    state = _registry[track_id]
    x1, y1, x2, y2 = box
    state.last_box   = box
    state.last_frame = frame_idx
    state.appearance = _appearance(frame, x1, y1, x2, y2)


def _on_track_lost(track_id: int):
    """Move a track to the ghost buffer when the tracker drops it."""
    if track_id in _registry:
        _ghosts[track_id] = _registry.pop(track_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Font
# ─────────────────────────────────────────────────────────────────────────────
_font_cache: dict = {}

def _font(size: int):
    if size not in _font_cache:
        for path in [
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                pass
        if size not in _font_cache:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


# ─────────────────────────────────────────────────────────────────────────────
#  Tag rendering
# ─────────────────────────────────────────────────────────────────────────────
PAD_X  = 16
PAD_Y  = 8
RADIUS = 12
LINE_C = (255, 255, 255, 150)


def _rounded_rect(draw, x0, y0, x1, y1, r, fill):
    draw.rectangle([x0+r, y0, x1-r, y1], fill=fill)
    draw.rectangle([x0, y0+r, x1, y1-r], fill=fill)
    for cx, cy in [(x0, y0), (x1-2*r, y0), (x0, y1-2*r), (x1-2*r, y1-2*r)]:
        draw.ellipse([cx, cy, cx+2*r, cy+2*r], fill=fill)


def render_tags(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    """detections: list of (track_id, x1, y1, x2, y2) in frame coords."""
    if not detections:
        return frame_bgr

    h, w   = frame_bgr.shape[:2]
    fsize  = max(20, int(h / 38))
    font   = _font(fsize)

    base    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for track_id, x1, y1, x2, y2 in detections:
        state          = _get_state(track_id)
        bg_rgba, tx_rgb = state.palette

        # Measure tag text
        tb     = font.getbbox(state.tag)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        cx       = (x1 + x2) // 2
        pill_w   = tw + 2 * PAD_X
        pill_h   = th + 2 * PAD_Y
        pill_x0  = int(max(4, min(cx - pill_w // 2, w - pill_w - 4)))
        pill_y0  = int(max(4, y1 - pill_h - 20))
        pill_x1  = pill_x0 + pill_w
        pill_y1  = pill_y0 + pill_h

        # Connector line + dot
        draw.line([(cx, y1), (cx, pill_y1)], fill=LINE_C, width=2)
        r = 5
        draw.ellipse([cx-r, y1-r, cx+r, y1+r], fill=(*tx_rgb, 230))

        # Pill + text
        _rounded_rect(draw, pill_x0, pill_y0, pill_x1, pill_y1, RADIUS, bg_rgba)
        draw.text((pill_x0 + PAD_X, pill_y0 + PAD_Y), state.tag,
                  font=font, fill=(*tx_rgb, 255))

        # Subtle person box
        draw.rectangle([x1, y1, x2, y2], outline=(*tx_rgb, 100), width=2)

    return cv2.cvtColor(np.array(Image.alpha_composite(base, overlay).convert("RGB")),
                        cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  3K upscale helper
# ─────────────────────────────────────────────────────────────────────────────
def scale_to_3k(frame: np.ndarray) -> np.ndarray:
    """Scale frame so width == OUT_WIDTH (3072), maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w == OUT_WIDTH:
        return frame
    new_w = OUT_WIDTH
    new_h = int(round(h * OUT_WIDTH / w))
    # Use INTER_LANCZOS4 for upscale (sharp), INTER_AREA for downscale (smooth)
    interp = cv2.INTER_LANCZOS4 if w < OUT_WIDTH else cv2.INTER_AREA
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def scale_boxes(detections, src_w, src_h, dst_w, dst_h):
    """Scale bounding boxes from detection resolution to output resolution."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    return [(tid,
             int(x1*sx), int(y1*sy),
             int(x2*sx), int(y2*sy))
            for tid, x1, y1, x2, y2 in detections]


# ─────────────────────────────────────────────────────────────────────────────
#  Video writer (H.264 via ffmpeg if available, mp4v fallback)
# ─────────────────────────────────────────────────────────────────────────────
def make_writer(path: Path, w: int, h: int) -> cv2.VideoWriter:
    for fourcc_str in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_str),
            OUT_FPS,
            (w, h),
        )
        if writer.isOpened():
            return writer
    raise RuntimeError("Could not open VideoWriter with any codec.")


# ─────────────────────────────────────────────────────────────────────────────
#  Core: process one video
# ─────────────────────────────────────────────────────────────────────────────
def process_video(input_path: Path, model: YOLO, conf: float) -> Path | None:
    _reset_state()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open {input_path.name}")
        return None

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output dimensions after 3K scaling
    out_h = int(round(src_h * OUT_WIDTH / src_w))
    out_w = OUT_WIDTH

    out_path = input_path.parent / f"tagged_{input_path.stem}.mp4"
    writer   = make_writer(out_path, out_w, out_h)

    # Frame step: skip frames if source fps > 24 so output is smooth 24fps
    frame_step = max(1, round(src_fps / OUT_FPS))

    print(f"  {input_path.name}")
    print(f"    Source  : {src_w}x{src_h} @ {src_fps:.1f}fps  ({total} frames)")
    print(f"    Output  : {out_w}x{out_h} @ {OUT_FPS}fps  → {out_path.name}")

    active_ids_prev: set[int] = set()
    frame_idx  = 0
    write_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth source frame to hit 24 fps output
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        # ── YOLO BoT-SORT tracking ──────────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            classes=[0],               # persons only
            conf=conf,
            verbose=False,
            tracker="botsort.yaml",    # ReID-aware tracker
        )

        detections      = []
        active_ids_curr = set()

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if box.id is None:
                    continue
                tid          = int(box.id[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                detections.append((tid, x1, y1, x2, y2))
                active_ids_curr.add(tid)
                _on_track_seen(tid, (x1, y1, x2, y2), frame, frame_idx)

        # Move dropped tracks to ghost buffer
        for lost_id in active_ids_prev - active_ids_curr:
            _on_track_lost(lost_id)
        active_ids_prev = active_ids_curr

        # ── Scale frame to 3K, scale boxes, render ──────────────────────────
        frame_3k       = scale_to_3k(frame)
        out_h_actual, out_w_actual = frame_3k.shape[:2]
        dets_scaled    = scale_boxes(detections, src_w, src_h,
                                     out_w_actual, out_h_actual)
        tagged         = render_tags(frame_3k, dets_scaled)

        writer.write(tagged)
        write_idx += 1

        frame_idx += 1
        if write_idx % 24 == 0:
            pct = frame_idx / total * 100 if total else 0
            print(f"    {write_idx} output frames  ({pct:.0f}%)", end="\r")

    cap.release()
    writer.release()
    print(f"\n  Saved → {out_path.name}  ({write_idx} frames @ {OUT_FPS}fps, {out_w}x{out_h})")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Overlay persistent life-status tags on every person in crowd footage."
    )
    parser.add_argument("videos", nargs="*",
                        help="Video files to process (default: all in ./webcamera/)")
    parser.add_argument("--conf",  type=float, default=0.30,
                        help="Detection confidence threshold (default 0.30)")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLO weights — yolov8n.pt (fast) or yolov8s.pt (accurate)")
    args = parser.parse_args()

    if args.videos:
        input_files = [Path(v) for v in args.videos]
    else:
        video_dir   = Path(__file__).parent / "webcamera"
        extensions  = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI"}
        input_files = sorted(
            f for f in video_dir.iterdir()
            if f.suffix in extensions and not f.stem.startswith("tagged_")
        )

    if not input_files:
        print("No videos found. Pass paths or put videos in the webcamera/ folder.")
        return

    print(f"Loading model : {args.model}")
    model = YOLO(args.model)

    print(f"Output        : {OUT_WIDTH}px wide, {OUT_FPS}fps")
    print(f"Videos        : {len(input_files)}\n")

    for video in input_files:
        process_video(video, model, conf=args.conf)

    print("\nAll done.")


if __name__ == "__main__":
    main()
