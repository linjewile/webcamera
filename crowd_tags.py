"""
crowd_tags.py
─────────────
Reads crowd video footage, detects and tracks every person using YOLOv8,
and overlays a persistent life-status tag above each person for as long
as they stay in the frame.

Usage:
    python crowd_tags.py                        # processes all videos in ./webcamera/
    python crowd_tags.py myvideo.mp4            # single file
    python crowd_tags.py *.mp4                  # glob

Output:
    tagged_<original_filename> saved alongside the source video.

Requirements:
    pip install ultralytics opencv-python pillow numpy
"""

import cv2
import numpy as np
import random
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


# ─────────────────────────────────────────────────────────────────────────────
#  Life-status tags  (add / remove freely)
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
]

# Colour palette: (background_rgba, text_rgb)
# Positive/neutral/negative sets — assigned randomly per person
PALETTES = [
    ((15,  15,  15,  200), (255, 255, 255)),   # near-black  / white
    ((255, 255, 255, 210), (15,  15,  15 )),   # white       / black
    ((20,  20,  40,  210), (180, 200, 255)),   # dark-blue   / soft-blue
    ((40,  10,  10,  210), (255, 180, 180)),   # dark-red    / soft-red
    ((10,  35,  20,  210), (150, 255, 180)),   # dark-green  / soft-green
    ((35,  25,  5,   210), (255, 210, 130)),   # dark-amber  / soft-amber
]


# ─────────────────────────────────────────────────────────────────────────────
#  Per-track state  (persists as long as the person is visible)
# ─────────────────────────────────────────────────────────────────────────────
class TrackState:
    def __init__(self):
        self.tag     = random.choice(TAGS)
        self.palette = random.choice(PALETTES)


track_registry: dict[int, TrackState] = {}

def get_state(track_id: int) -> TrackState:
    if track_id not in track_registry:
        track_registry[track_id] = TrackState()
    return track_registry[track_id]


# ─────────────────────────────────────────────────────────────────────────────
#  Font loading
# ─────────────────────────────────────────────────────────────────────────────
def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
#  Tag rendering
# ─────────────────────────────────────────────────────────────────────────────
PAD_X = 14   # horizontal padding inside the tag pill
PAD_Y = 7    # vertical padding inside the tag pill
RADIUS = 10  # pill corner radius
LINE_COLOR = (255, 255, 255, 160)  # connector line colour

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def get_font(size: int):
    if size not in _font_cache:
        _font_cache[size] = load_font(size)
    return _font_cache[size]


def draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill):
    x0, y0, x1, y1 = xy
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    draw.ellipse([x0, y0, x0 + 2*radius, y0 + 2*radius], fill=fill)
    draw.ellipse([x1 - 2*radius, y0, x1, y0 + 2*radius], fill=fill)
    draw.ellipse([x0, y1 - 2*radius, x0 + 2*radius, y1], fill=fill)
    draw.ellipse([x1 - 2*radius, y1 - 2*radius, x1, y1], fill=fill)


def render_tags(frame_bgr: np.ndarray, detections: list[tuple]) -> np.ndarray:
    """
    detections: list of (track_id, x1, y1, x2, y2)
    Returns a new frame with tags composited.
    """
    if not detections:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    font_size = max(16, int(h / 40))  # scales with video resolution
    font = get_font(font_size)

    # Work in RGBA for semi-transparent compositing
    base   = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for track_id, x1, y1, x2, y2 in detections:
        state = get_state(track_id)
        bg_rgba, text_rgb = state.palette

        # Measure text
        bbox  = font.getbbox(state.tag)
        tw    = bbox[2] - bbox[0]
        th    = bbox[3] - bbox[1]

        # Tag pill position — centred above the bounding box top
        cx       = (x1 + x2) // 2
        pill_w   = tw + 2 * PAD_X
        pill_h   = th + 2 * PAD_Y
        pill_x0  = max(4, min(cx - pill_w // 2, w - pill_w - 4))
        pill_y0  = max(4, y1 - pill_h - 18)
        pill_x1  = pill_x0 + pill_w
        pill_y1  = pill_y0 + pill_h

        # Connector line: top-center of box → bottom of pill
        lx = cx
        draw.line([(lx, y1), (lx, pill_y1)], fill=LINE_COLOR, width=2)

        # Dot on bounding box top
        dot_r = 4
        draw.ellipse([lx - dot_r, y1 - dot_r, lx + dot_r, y1 + dot_r],
                     fill=(*text_rgb, 220))

        # Pill background
        draw_rounded_rect(draw, (pill_x0, pill_y0, pill_x1, pill_y1),
                          RADIUS, bg_rgba)

        # Text
        draw.text((pill_x0 + PAD_X, pill_y0 + PAD_Y),
                  state.tag, font=font, fill=(*text_rgb, 255))

        # Thin bounding box on the person (subtle, same text colour)
        draw.rectangle([x1, y1, x2, y2],
                       outline=(*text_rgb, 120), width=2)

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  Core processing
# ─────────────────────────────────────────────────────────────────────────────
def process_video(input_path: Path, model: YOLO, conf: float = 0.35) -> Path:
    track_registry.clear()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"  [!] Could not open {input_path.name}")
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = input_path.parent / f"tagged_{input_path.stem}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"  Processing: {input_path.name}  ({width}x{height} @ {fps:.1f}fps, {total} frames)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 tracking — persist=True keeps IDs consistent across frames
        results = model.track(
            frame,
            persist=True,
            classes=[0],        # person only
            conf=conf,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append((track_id, x1, y1, x2, y2))

        tagged_frame = render_tags(frame, detections)
        writer.write(tagged_frame)

        frame_idx += 1
        if frame_idx % 60 == 0:
            pct = frame_idx / total * 100 if total else 0
            print(f"    {frame_idx}/{total} frames  ({pct:.0f}%)", end="\r")

    cap.release()
    writer.release()
    print(f"\n  Done → {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Overlay persistent life-status tags on every person in crowd footage."
    )
    parser.add_argument(
        "videos", nargs="*",
        help="Video file(s) to process. If omitted, processes all videos in ./webcamera/"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="YOLO detection confidence threshold (default 0.35)"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLO model weights (default yolov8n.pt). Use yolov8s.pt for better accuracy."
    )
    args = parser.parse_args()

    # Resolve input files
    if args.videos:
        input_files = [Path(v) for v in args.videos]
    else:
        video_dir  = Path(__file__).parent / "webcamera"
        extensions = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"}
        input_files = [
            f for f in video_dir.iterdir()
            if f.suffix in extensions and not f.stem.startswith("tagged_")
        ]

    if not input_files:
        print("No video files found. Pass a path or put videos in the webcamera/ folder.")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"\nProcessing {len(input_files)} video(s)...\n")
    for video in input_files:
        process_video(video, model, conf=args.conf)

    print("\nAll done.")


if __name__ == "__main__":
    main()
