"""
crowd_tags.py
─────────────
Reads crowd video footage, detects and tracks every person using YOLOv8,
and overlays a persistent life-status tag above each person for as long
as they stay in the frame.

PersonHashMap
─────────────
Every unique person detected gets a fingerprint derived from a quantised
HSV histogram of their body crop.  The map  fingerprint → quote  is saved
to  person_tags.json  next to this script so the same person across
different video files always receives the same quote.

The JSON file is human-readable — you can open it and edit any quote
before re-running to customise what specific people say.

Re-identification layers
─────────────────────────
  1. BoT-SORT      — Kalman + ReID features (frame-to-frame)
  2. Ghost buffer  — remembers lost tracks for ~3 s; matches by appearance
                     + last position when the person re-enters the frame
  3. PersonHashMap — persistent cross-video identity via appearance hash

Export: 3 072 px wide · 24 fps · H.264

Usage:
    python crowd_tags.py                      # all videos in ./webcamera/
    python crowd_tags.py myvideo.mp4          # single file
    --conf  0.28   lower = catch more people in dense crowds
    --model yolov8s.pt   better accuracy (slower)
"""

import cv2
import json
import numpy as np
import random
import argparse
import time
import threading
import queue
from pathlib import Path
from ultralytics import YOLO
import torchvision.ops as _tv_ops
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    import supervision as sv
    _SV = True
except ImportError:
    _SV = False

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn,
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box as rbox
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None

# ─────────────────────────────────────────────────────────────────────────────
#  Export  (output FPS now matches source — set dynamically in process_video)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID / ghost buffer
# ─────────────────────────────────────────────────────────────────────────────
GHOST_FRAMES    = 72
REID_APP_W      = 0.65
REID_IOU_W      = 0.35
REID_MIN_SCORE  = 0.42

# ─────────────────────────────────────────────────────────────────────────────
#  Frame-skip: run YOLO every N-th frame, reuse detections in between
# ─────────────────────────────────────────────────────────────────────────────
DETECT_EVERY    = 4   # run detection/tracking every 4th frame; render cached boxes otherwise

# ─────────────────────────────────────────────────────────────────────────────
#  Tiled inference  (SAHI-style)
#  Slices the frame into overlapping tiles so small/distant people are caught.
#  Each tile is run through YOLO independently, boxes are mapped back to the
#  full-frame coordinate space, then merged with NMS.
# ─────────────────────────────────────────────────────────────────────────────
TILE_SIZE    = 640   # each tile fed to YOLO (px)
TILE_OVERLAP = 0.25  # 25 % overlap between adjacent tiles
TILE_NMS_IOU = 0.45  # IoU threshold for merging duplicate boxes across tiles


def _tile_detect(model: YOLO, frame: np.ndarray,
                 conf: float, imgsz: int = TILE_SIZE) -> list[tuple]:
    """
    Slice `frame` into overlapping tiles, run YOLO on each, map boxes back
    to full-frame coordinates, then merge duplicates with NMS.

    Returns list of (x1, y1, x2, y2, score) in full-frame pixel space.
    """
    H, W = frame.shape[:2]
    stride = int(imgsz * (1 - TILE_OVERLAP))
    all_boxes:  list[list[float]] = []
    all_scores: list[float]       = []

    # Generate tile origins
    xs = list(range(0, max(1, W - imgsz), stride)) + [max(0, W - imgsz)]
    ys = list(range(0, max(1, H - imgsz), stride)) + [max(0, H - imgsz)]
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    for y0 in ys:
        for x0 in xs:
            x1c = min(x0 + imgsz, W)
            y1c = min(y0 + imgsz, H)
            tile = frame[y0:y1c, x0:x1c]

            results = model.predict(tile, classes=[0], conf=conf,
                                    verbose=False, imgsz=imgsz)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    # Map back to full-frame coords
                    all_boxes.append([bx1 + x0, by1 + y0,
                                      bx2 + x0, by2 + y0])
                    all_scores.append(score)

    if not all_boxes:
        return []

    boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    keep     = _tv_ops.nms(boxes_t, scores_t, TILE_NMS_IOU)

    return [(int(boxes_t[i][0]), int(boxes_t[i][1]),
             int(boxes_t[i][2]), int(boxes_t[i][3]),
             float(scores_t[i])) for i in keep.tolist()]

# ─────────────────────────────────────────────────────────────────────────────
#  Life-status tags
# ─────────────────────────────────────────────────────────────────────────────
TAGS = [
    "just got fired",
    "got a promotion today",
    "served in the military",
    "secretly a millionaire",
    "owes $167k in student loans",
    "just asked for a raise",
    "hasn't filed taxes in 3 years",
    "just quit their job",
    "1 week sober",
    "lost their wallet",
    "is going to be late for work",
    "just lost their mom",
    "first date tonight",
    "very hungover",
    "struggling with depression",
    "getting married next week",
    "texting their ex",
    "was ghosted recently",
    "diassociating",
    "is going to the army soon",
    "just got a tattoo",
    "very high right now",
    "just farted",
    "phone is at 1%",
    "just got out of a toxic relationship",
    "just had a baby",
    "sibling got deported",
    "paycheck just hit",
    "hasn't slept in 2 days",
    "just found $20 on the ground",
    "just found out they're pregnant",
    "going through a breakup",
    "just got off a 12hr shift",
    "running late",
    "forgot someone's birthday",
    "just moved to a new city",
    "struggling with their mental health ",
    "best day of their life",
    "worst week of their life",
    "just got diagnosed with cancer",
    "thinking about changing careers",
    "lost their life savings in crypto",
    "just graduated",
    "their parlay just hit",
    "almost went pro but tore their ACL",
    "hasn't eaten all day",
    "just had their 4th abortion",
    "on their 4th coffee",
    "pretending to be fine",
    "was left on read",
    "taxes just hit",
    "mid-life crisis incoming",
    "today is their birthday",
    "grieving ",
]

PALETTES = [
    ((15,  15,  15,  210), (255, 255, 255)),
    ((245, 245, 245, 215), (20,  20,  20 )),
    ((18,  22,  45,  215), (160, 190, 255)),
    ((42,  10,  10,  215), (255, 170, 170)),
    ((8,   38,  18,  215), (130, 255, 170)),
    ((38,  26,  4,   215), (255, 210, 120)),
]


# ─────────────────────────────────────────────────────────────────────────────
#  PersonHashMap
#  Maps appearance fingerprint → {"quote": str, "palette": int, "seen": int}
#  Persisted to person_tags.json so it survives across sessions.
# ─────────────────────────────────────────────────────────────────────────────
HASHMAP_PATH     = Path(__file__).parent / "person_tags.json"
FINGERPRINT_BINS = (12, 5)    # hue bins, saturation bins  → 60-value vector
MATCH_THRESHOLD  = 0.88       # cosine similarity to call two fingerprints the same person


class PersonHashMap:
    """
    Stores  fingerprint_hex → {quote, palette_idx, seen_count}  in JSON.

    Fingerprint
    ───────────
    Computed from the top-60% of a person's bounding box (torso — more
    stable than legs or head).  A 12×5 HSV histogram is computed,
    L2-normalised, then each value is quantised to a 2-digit hex byte
    and concatenated into a ~120-char hex string.

    Matching
    ────────
    Cosine similarity between two float32 histogram vectors.
    If similarity ≥ MATCH_THRESHOLD the person is considered the same.
    When multiple stored entries match, the closest one wins.
    """

    def __init__(self, path: Path = HASHMAP_PATH):
        self.path    = path
        self._map: dict[str, dict] = {}        # hex_key → record
        self._vecs: dict[str, np.ndarray] = {} # hex_key → float32 vector (cached)
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._map = json.load(f)
                # Rebuild float vector cache
                for key, rec in self._map.items():
                    if "vec" in rec:
                        self._vecs[key] = np.array(rec["vec"], dtype=np.float32)
            except (json.JSONDecodeError, KeyError):
                self._map = {}
        print(f"  PersonHashMap: {len(self._map)} known person(s) loaded from {self.path.name}")

    def save(self):
        # Embed the float vector in the JSON so we can reconstruct on reload
        for key, rec in self._map.items():
            if key in self._vecs:
                rec["vec"] = self._vecs[key].tolist()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._map, f, indent=2, ensure_ascii=False)

    def print_map(self):
        """Pretty-print the full hashmap to the console."""
        print(f"\n{'─'*60}")
        print(f"  PersonHashMap  ({len(self._map)} entries)  →  {self.path.name}")
        print(f"{'─'*60}")
        for i, (key, rec) in enumerate(self._map.items(), 1):
            seen = rec.get("seen", 1)
            print(f"  {i:>3}.  {key[:24]}...  "
                  f"seen:{seen:>4}x  \"{rec['quote']}\"")
        print(f"{'─'*60}\n")

    # ── fingerprinting ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_vec(crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Returns an L2-normalised float32 vector from the torso portion
        of a person crop.  Returns None if the crop is too small.
        """
        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 10:
            return None

        # Use only the top 60% (torso — clothing is stable; legs vary)
        torso = crop_bgr[:int(h * 0.6), :]

        hsv  = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [FINGERPRINT_BINS[0], FINGERPRINT_BINS[1]],
            [0, 180, 0, 256],
        )
        vec = hist.flatten().astype(np.float32)

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return None
        return vec / norm

    @staticmethod
    def _vec_to_key(vec: np.ndarray) -> str:
        """Quantise each float to 0-255 and encode as hex string."""
        quantised = np.clip(vec * 255, 0, 255).astype(np.uint8)
        return quantised.tobytes().hex()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # both already L2-normalised

    # ── public API ───────────────────────────────────────────────────────────

    def get_quote(self, crop_bgr: np.ndarray) -> tuple[str, int] | None:
        """
        Return (quote, palette_idx) for this crop, or None if crop too small.
        Tag is ALWAYS drawn fresh from the shuffled deck — never reused from
        a previous session — to guarantee no two people in the same video
        share a tag.
        """
        vec = self._compute_vec(crop_bgr)
        if vec is None:
            return None

        # Vectorised cosine search (all vecs are L2-normalised)
        best_key = None
        if self._vecs:
            keys = list(self._vecs.keys())
            mat  = np.stack([self._vecs[k] for k in keys])  # (N, D)
            sims = mat @ vec                                 # (N,)
            idx  = int(np.argmax(sims))
            if sims[idx] >= MATCH_THRESHOLD:
                best_key = keys[idx]

        if best_key is not None:
            avg = self._vecs[best_key] * 0.9 + vec * 0.1
            avg /= (np.linalg.norm(avg) + 1e-9)
            self._vecs[best_key] = avg
            self._map[best_key]["seen"] = self._map[best_key].get("seen", 1) + 1
            pal = self._map[best_key]["palette_idx"]
        else:
            new_key = self._vec_to_key(vec)
            pal = random.randrange(len(PALETTES))
            self._map[new_key] = {
                "quote":       "",
                "palette_idx": pal,
                "seen":        1,
            }
            self._vecs[new_key] = vec

        # Always draw a fresh unique tag from the deck
        tag = _draw_unique_tag()
        return tag, pal


# ─────────────────────────────────────────────────────────────────────────────
#  Track state + ghost buffer  (per-video)
# ─────────────────────────────────────────────────────────────────────────────
SMOOTH_ALPHA = 0.3   # EMA factor: lower = smoother/slower, higher = more responsive

class TrackState:
    __slots__ = ("tag", "palette", "last_box", "smooth_box", "last_frame",
                 "appearance", "_app_ctr")

    def __init__(self, tag: str, palette_idx: int):
        self.tag        = tag
        self.palette    = PALETTES[palette_idx]
        self.last_box   = None
        self.smooth_box = None   # EMA-smoothed (x1,y1,x2,y2)
        self.last_frame = 0
        self.appearance = None
        self._app_ctr   = 0

    def update_smooth_box(self, box: tuple):
        """Blend new raw box into the smoothed position."""
        if self.smooth_box is None:
            self.smooth_box = tuple(float(c) for c in box)
        else:
            a = SMOOTH_ALPHA
            self.smooth_box = tuple(
                a * new + (1 - a) * old
                for new, old in zip(box, self.smooth_box)
            )


_registry: dict[int, TrackState] = {}
_ghosts:   dict[int, TrackState] = {}
_last_frame_count: int = 0
_tag_deck:  list[str] = []         # shuffled deck — pop to assign, never put back


def _reset_state():
    _registry.clear()
    _ghosts.clear()
    _tag_deck.clear()
    _tag_deck.extend(TAGS)
    random.shuffle(_tag_deck)


def _get_state(track_id: int) -> TrackState:
    return _registry[track_id]


def _draw_unique_tag() -> str:
    """Pop the next tag from the shuffled deck. Like drawing a card — once
    it's drawn, it's gone. No repeats until the deck is exhausted."""
    if _tag_deck:
        return _tag_deck.pop()
    # Deck empty (45+ people) — impossible to avoid reuse, but make it
    # obvious by appending a number so it's visually distinct.
    _tag_deck.extend(TAGS)
    random.shuffle(_tag_deck)
    return _tag_deck.pop()


# ── appearance helpers ────────────────────────────────────────────────────────

def _appearance_hist(frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray | None:
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
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
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _expire_ghosts(current_frame: int):
    dead = [k for k, gs in _ghosts.items()
            if current_frame - gs.last_frame > GHOST_FRAMES]
    for k in dead:
        del _ghosts[k]


def _match_ghost(box, app, current_frame: int) -> int | None:
    _expire_ghosts(current_frame)
    best_id, best_score = None, REID_MIN_SCORE
    for gid, gs in _ghosts.items():
        if gs.last_box is None:
            continue
        score = REID_APP_W * _app_sim(app, gs.appearance) + REID_IOU_W * _iou(box, gs.last_box)
        if score > best_score:
            best_score, best_id = score, gid
    return best_id


def _on_track_seen(track_id: int, box, frame: np.ndarray,
                   frame_idx: int, hashmap: PersonHashMap):
    x1, y1, x2, y2 = box

    if track_id not in _registry:
        app  = _appearance_hist(frame, x1, y1, x2, y2)
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        ghost = _match_ghost(box, app, frame_idx)

        if ghost is not None:
            _registry[track_id] = _ghosts.pop(ghost)
        else:
            result = hashmap.get_quote(crop) if crop.size > 0 else None
            if result:
                tag, pal = result
            else:
                tag, pal = _draw_unique_tag(), random.randrange(len(PALETTES))
            _registry[track_id] = TrackState(tag, pal)
        state = _registry[track_id]
        state.appearance = app
    else:
        state = _registry[track_id]
        state._app_ctr += 1
        # Only refresh appearance every 5th frame for established tracks
        if state._app_ctr % 5 == 0:
            state.appearance = _appearance_hist(frame, x1, y1, x2, y2)

    state.last_box   = box
    state.update_smooth_box(box)
    state.last_frame = frame_idx


def _on_track_lost(track_id: int):
    if track_id in _registry:
        _ghosts[track_id] = _registry.pop(track_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Tag rendering  (Pillow + custom or system font)
# ─────────────────────────────────────────────────────────────────────────────
_LOCAL_FONT = Path(__file__).parent / "font familia" / "Butler FREE" \
              / "OTF - best in most cases" / "Butler-Free-Bd.otf"
_FONT_PATH = str(_LOCAL_FONT) if _LOCAL_FONT.exists() else "arial.ttf"
PAD_X = 8
PAD_Y = 4
_MIN_FONT_SIZE = 10
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_FONT_PATH, size)
    return _font_cache[size]


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [text]


def _fit_font_size_wrapped(text: str, max_width: int, max_height: int) -> tuple[int, list[str]]:
    """Find the largest font size where the wrapped text fits in the box."""
    lo, hi = _MIN_FONT_SIZE, 200
    best_size = lo
    best_lines = [text]
    for _ in range(12):
        mid = (lo + hi) // 2
        font = _get_font(mid)
        lines = _wrap_text(text, font, max_width - 2 * PAD_X)
        # Measure total height
        line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
        total_h = line_h * len(lines) + PAD_Y * (len(lines) - 1)
        # Check widest line fits
        widest = max((font.getbbox(ln)[2] - font.getbbox(ln)[0]) for ln in lines)
        if widest + 2 * PAD_X <= max_width and total_h + 2 * PAD_Y <= max_height:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1
    return best_size, best_lines


def render_tags(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    if not detections:
        return frame_bgr

    h, w = frame_bgr.shape[:2]

    # Convert to PIL once, draw all tags, convert back
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    for track_id, x1, y1, x2, y2 in detections:
        if track_id not in _registry:
            continue
        state = _get_state(track_id)

        # Use smoothed box for stable text positioning
        if state.smooth_box is not None:
            sx1, sy1, sx2, sy2 = (int(c) for c in state.smooth_box)
        else:
            sx1, sy1, sx2, sy2 = x1, y1, x2, y2

        box_w = sx2 - sx1
        box_h = sy2 - sy1
        if box_w < 10 or box_h < 10:
            continue

        font_size, lines = _fit_font_size_wrapped(state.tag, box_w, box_h)
        font = _get_font(font_size)
        line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
        line_spacing = PAD_Y
        total_text_h = line_h * len(lines) + line_spacing * (len(lines) - 1)

        # Centre the text block on the smoothed person position
        cx = (sx1 + sx2) // 2
        cy = (sy1 + sy2) // 2
        block_y = cy - total_text_h // 2

        for i, line in enumerate(lines):
            lbbox = font.getbbox(line)
            lw = lbbox[2] - lbbox[0]
            text_x = int(max(0, min(cx - lw // 2, w - lw))) - lbbox[0]
            text_y = int(block_y + i * (line_h + line_spacing)) - lbbox[1]
            # Dark outline for readability on any background
            outline_d = max(1, font_size // 18)
            for ox, oy in [(-outline_d,0),(outline_d,0),(0,-outline_d),(0,outline_d),
                           (-outline_d,-outline_d),(outline_d,-outline_d),
                           (-outline_d,outline_d),(outline_d,outline_d)]:
                draw.text((text_x+ox, text_y+oy), line, fill=(0, 0, 0), font=font)
            draw.text((text_x, text_y), line, fill=(255, 255, 255), font=font)

    frame_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame_bgr


# ─────────────────────────────────────────────────────────────────────────────
#  3K scaling + writer
# ─────────────────────────────────────────────────────────────────────────────
def scale_boxes(dets, sw, sh, dw, dh):
    sx, sy = dw / sw, dh / sh
    return [(t, int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
            for t, x1, y1, x2, y2 in dets]


def make_writer(path: Path, w: int, h: int, fps: float = 30.0) -> cv2.VideoWriter:
    for fc in ("mp4v", "avc1", "H264"):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fc), fps, (w, h))
        if wr.isOpened():
            return wr
    raise RuntimeError("Could not open VideoWriter.")


# ─────────────────────────────────────────────────────────────────────────────
#  Process one video
# ─────────────────────────────────────────────────────────────────────────────
def process_video(input_path: Path, model: YOLO,
                  conf: float, hashmap: PersonHashMap,
                  tiled: bool = False,
                  imgsz: int = 640,
                  augment: bool = False,
                  half: bool = False,
                  detect_every: int = DETECT_EVERY) -> Path | None:
    global _last_frame_count
    _reset_state()

    # Per-video tracker for tiled mode
    sv_tracker = sv.ByteTrack() if (tiled and _SV) else None

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        (_console.print(f"[red]  [!] Cannot open {input_path.name}[/red]") if _RICH
         else print(f"  [!] Cannot open {input_path.name}"))
        return None

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = src_fps                       # output matches source — no frame dropping

    out_path = input_path.parent / f"tagged_{input_path.stem}.mp4"
    writer   = make_writer(out_path, src_w, src_h, out_fps)

    total_out   = max(1, total)

    if _RICH:
        _console.print(f"  [dim]Source :[/dim] {src_w}×{src_h} @ {src_fps:.1f}fps  "
                       f"([dim]{total} frames[/dim])")
        _console.print(f"  [dim]Output :[/dim] [cyan]{src_w}×{src_h}[/cyan] "
                       f"@ [cyan]{out_fps:.1f}fps[/cyan]  [dim](matches source)[/dim]")
    else:
        print(f"  Source : {src_w}x{src_h} @ {src_fps:.1f}fps  ({total} frames)")
        print(f"  Output : {src_w}x{src_h} @ {out_fps:.1f}fps  (matches source)")

    prev_ids: set[int] = set()
    frame_idx = write_idx = 0
    cached_detections: list = []   # reused on skipped frames

    # ── Read-ahead queue ──────────────────────────────────────────────────────
    # A background thread pre-decodes frames so YOLO never stalls on I/O.
    # Queue holds (frame_idx, frame_bgr); sentinel None signals end of stream.
    _Q_SIZE = 16
    _frame_q: queue.Queue = queue.Queue(maxsize=_Q_SIZE)

    def _reader():
        idx = 0
        while True:
            ret, frm = cap.read()
            if not ret:
                _frame_q.put(None)
                break
            _frame_q.put((idx, frm))   # every frame — no skipping
            idx += 1
    _reader_thread = threading.Thread(target=_reader, daemon=True)
    _reader_thread.start()

    # ── Write-behind queue ────────────────────────────────────────────────────
    # A background thread encodes/writes frames so the main loop never blocks
    # on VideoWriter I/O.  Sentinel None signals end of stream.
    _write_q: queue.Queue = queue.Queue(maxsize=32)

    def _writer_fn():
        while True:
            f = _write_q.get()
            if f is None:
                break
            writer.write(f)
    _writer_thread = threading.Thread(target=_writer_fn, daemon=True)
    _writer_thread.start()

    progress = _make_progress()
    task_id  = None
    ctx      = progress if _RICH else None

    def _run():
        nonlocal frame_idx, write_idx, prev_ids

        while True:
            item = _frame_q.get()
            if item is None:
                break
            frame_idx, frame = item

            # ── Frame-skip: reuse cached detections on non-detect frames ──
            run_detect = (write_idx % detect_every == 0)

            if not run_detect:
                tagged = render_tags(frame, cached_detections)
                _write_q.put(tagged)
                write_idx += 1
                if _RICH and task_id is not None:
                    progress.update(task_id, advance=1,
                                    description=f"[white]{input_path.stem[:30]}[/white]"
                                                f"  [dim yellow]{len(cached_detections)} people[/dim yellow]")
                elif write_idx % 24 == 0:
                    pct = frame_idx / total * 100 if total else 0
                    print(f"    {write_idx} frames written  ({pct:.0f}%)", end="\r")
                continue

            if tiled and _SV:
                # ── Tiled inference + supervision ByteTrack ──────────────────
                # Tile detect gives us far more boxes for small/distant people.
                # supervision.ByteTrack accepts pre-computed boxes directly,
                # so track state is correctly maintained across frames.
                raw_boxes = _tile_detect(model, frame, conf)
                if raw_boxes:
                    xyxy   = np.array([[x1,y1,x2,y2] for x1,y1,x2,y2,_ in raw_boxes])
                    confs  = np.array([s for _,_,_,_,s in raw_boxes])
                    clsids = np.zeros(len(raw_boxes), dtype=int)
                    sv_det = sv.Detections(xyxy=xyxy, confidence=confs, class_id=clsids)
                else:
                    sv_det = sv.Detections.empty()

                sv_tracks = sv_tracker.update_with_detections(sv_det)

                detections = []
                curr_ids   = set()
                for i in range(len(sv_tracks)):
                    tid          = int(sv_tracks.tracker_id[i])
                    x1,y1,x2,y2 = map(int, sv_tracks.xyxy[i])
                    detections.append((tid, x1, y1, x2, y2))
                    curr_ids.add(tid)
                    _on_track_seen(tid, (x1, y1, x2, y2), frame, frame_idx, hashmap)

            else:
                # ── Standard model.track at high resolution ──────────────────
                # imgsz=1280 catches small/distant people that 640 misses.
                # iou=0.4 separates closely-packed people better than default 0.7.
                results = model.track(
                    frame, persist=True, classes=[0],
                    conf=conf, iou=0.4, imgsz=imgsz,
                    verbose=False, tracker="botsort.yaml",
                    augment=augment, half=half,
                )

                detections = []
                curr_ids   = set()

                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        if box.id is None:
                            continue
                        tid          = int(box.id[0])
                        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                        detections.append((tid, x1, y1, x2, y2))
                        curr_ids.add(tid)
                        _on_track_seen(tid, (x1, y1, x2, y2), frame, frame_idx, hashmap)

            for lost in prev_ids - curr_ids:
                _on_track_lost(lost)
            prev_ids = curr_ids

            cached_detections = detections   # cache for skipped frames
            tagged = render_tags(frame, detections)
            _write_q.put(tagged)

            write_idx += 1

            if _RICH and task_id is not None:
                n_people = len(curr_ids)
                progress.update(task_id, advance=1,
                                description=f"[white]{input_path.stem[:30]}[/white]"
                                            f"  [dim yellow]{n_people} people[/dim yellow]")
            elif write_idx % 24 == 0:
                pct = frame_idx / total * 100 if total else 0
                print(f"    {write_idx} frames written  ({pct:.0f}%)", end="\r")

    if _RICH:
        with progress:
            task_id = progress.add_task(
                f"[white]{input_path.stem[:30]}[/white]",
                total=total_out,
            )
            _run()
    else:
        _run()

    _write_q.put(None)
    _writer_thread.join()
    _reader_thread.join()
    cap.release()
    writer.release()
    hashmap.save()
    _last_frame_count = write_idx

    if _RICH:
        _console.print(f"  [green]✓[/green] Saved → [bold]{out_path.name}[/bold]  "
                       f"([cyan]{write_idx}[/cyan] frames)\n")
    else:
        print(f"\n  Saved → {out_path.name}  ({write_idx} frames)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def _print_banner():
    if _RICH:
        title = Text("CROWD TAGS", style="bold white")
        sub   = Text("life-status overlay engine", style="dim white")
        panel = Panel.fit(
            f"[bold white]CROWD TAGS[/bold white]\n[dim]life-status overlay engine[/dim]",
            border_style="bright_yellow",
            padding=(1, 4),
        )
        _console.print()
        _console.print(panel, justify="center")
        _console.print()
    else:
        print("\n" + "═"*50)
        print("   CROWD TAGS  —  life-status overlay engine")
        print("═"*50 + "\n")


def _print_summary(results: list[tuple[str, int, float]]):
    """results: list of (filename, frame_count, elapsed_secs)"""
    if _RICH:
        table = Table(title="[bold]Processing Summary[/bold]",
                      box=rbox.SIMPLE_HEAVY, border_style="bright_yellow",
                      show_lines=False)
        table.add_column("File",    style="white")
        table.add_column("Frames",  style="cyan",  justify="right")
        table.add_column("Time",    style="yellow", justify="right")
        for name, frames, elapsed in results:
            mins, secs = divmod(int(elapsed), 60)
            table.add_row(name, str(frames), f"{mins}m {secs:02d}s")
        _console.print()
        _console.print(table)
        _console.print("[bold bright_yellow]All done.[/bold bright_yellow]\n")
    else:
        print("\nSummary:")
        for name, frames, elapsed in results:
            mins, secs = divmod(int(elapsed), 60)
            print(f"  {name}: {frames} frames  {mins}m {secs:02d}s")
        print("All done.")


def _make_progress():
    if _RICH:
        return Progress(
            SpinnerColumn(spinner_name="dots", style="bright_yellow"),
            TextColumn("[bold white]{task.description}"),
            BarColumn(bar_width=40, style="yellow", complete_style="bright_yellow"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_console,
            refresh_per_second=10,
        )
    return None


def _prompt_for_videos() -> list[Path]:
    """
    Prompt the user to paste/drag-drop a single video file path.
    """
    exts = {".mp4", ".mov", ".avi", ".mkv"}

    if _RICH:
        _console.print("[dim]Paste or drag a video file path below.[/dim]\n")
    else:
        print("Paste or drag a video file path below.\n")

    while True:
        try:
            line = input("  Video path: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print()
            return []
        if not line:
            return []
        p = Path(line)
        if not p.exists():
            print(f"    [!] File not found: {p}")
            continue
        if p.suffix.lower() not in exts:
            print(f"    [!] Unsupported format: {p.suffix}  (use mp4/mov/avi/mkv)")
            continue
        print(f"    ✓ Loaded: {p.name}")
        return [p]


def main():
    parser = argparse.ArgumentParser(
        description="Overlay persistent life-status tags on crowd footage."
    )
    parser.add_argument("videos", nargs="*",
                        help="Video files (default: all in ./webcamera/)")
    parser.add_argument("--conf",   type=float, default=0.18,
                        help="Detection confidence threshold (default 0.18)")
    parser.add_argument("--model",  default="yolov8m.pt",
                        help="YOLO model weights (default yolov8m — best for crowds)")
    parser.add_argument("--imgsz",  type=int, default=1280,
                        help="Input resolution for YOLO (default 1280 — catches small/distant people)")
    parser.add_argument("--tile",   action="store_true",
                        help="Tiled inference: slice frame into overlapping patches "
                             "(best for very dense crowds; requires `pip install supervision`)")
    parser.add_argument("--augment", action="store_true",
                        help="Test-time augmentation — slower but higher recall")
    parser.add_argument("--half", action="store_true",
                        help="fp16 inference — ~2x faster on NVIDIA GPU (CUDA only)")
    parser.add_argument("--skip", type=int, default=DETECT_EVERY,
                        help=f"Run detection every N-th frame, reuse boxes in between (default {DETECT_EVERY})")
    parser.add_argument("--print-map", action="store_true",
                        help="Print the full person→quote hashmap and exit")
    args = parser.parse_args()

    if args.tile and not _SV:
        print("WARNING: --tile requires `pip install supervision`. Falling back to standard mode.")
        args.tile = False

    # Auto-enable fp16 on CUDA for a ~2x speedup
    if not args.half and torch.cuda.is_available():
        args.half = True

    _print_banner()

    hashmap = PersonHashMap()

    if args.print_map:
        hashmap.print_map()
        return

    # ── Collect video file ────────────────────────────────────────────────
    if args.videos:
        input_files = [Path(args.videos[0])]
    else:
        input_files = _prompt_for_videos()

    if not input_files:
        (_console.print("[yellow]No videos found.[/yellow]") if _RICH
         else print("No videos found."))
        return

    # ── Load model with spinner ───────────────────────────────────────────────
    if _RICH:
        with _console.status(f"[bright_yellow]Loading {args.model}…[/bright_yellow]",
                             spinner="dots"):
            model = YOLO(args.model)
        if args.half:
            model.half()
        _console.print(f"[green]✓[/green] Model loaded: [bold]{args.model}[/bold]"
                       + (" [dim](fp16)[/dim]" if args.half else ""))
    else:
        print(f"Loading {args.model}…")
        model = YOLO(args.model)
        if args.half:
            model.half()

    mode_str = ("tiled" if args.tile else f"imgsz={args.imgsz}") + \
               (" +augment" if args.augment else "")
    if _RICH:
        _console.print(f"  Output  : [cyan]source resolution[/cyan] · "
                       f"[cyan]native fps[/cyan]")
        _console.print(f"  Mode    : [cyan]{mode_str}[/cyan]  "
                       f"conf=[cyan]{args.conf}[/cyan]")
        _console.print(f"  Videos  : [cyan]{len(input_files)}[/cyan]\n")
    else:
        print(f"Output : source resolution · native fps")
        print(f"Mode   : {mode_str}  conf={args.conf}")
        print(f"Videos : {len(input_files)}\n")

    # ── Process each video ────────────────────────────────────────────────────
    summary = []
    for i, video in enumerate(input_files, 1):
        if _RICH:
            _console.rule(f"[bold white]{i}/{len(input_files)}  {video.name}[/bold white]")
        t0 = time.time()
        out = process_video(video, model, conf=args.conf, hashmap=hashmap,
                            tiled=args.tile, imgsz=args.imgsz,
                            augment=args.augment, half=args.half,
                            detect_every=args.skip)
        elapsed = time.time() - t0
        # process_video returns frame count via its own print; we track externally
        # Collect frame count from out metadata isn't available, use a sentinel
        summary.append((video.name, _last_frame_count, elapsed))

    hashmap.print_map()
    _print_summary(summary)


if __name__ == "__main__":
    main()
