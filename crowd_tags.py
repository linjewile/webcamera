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
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
#  Export
# ─────────────────────────────────────────────────────────────────────────────
OUT_WIDTH = 3072
OUT_FPS   = 24

# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID / ghost buffer
# ─────────────────────────────────────────────────────────────────────────────
GHOST_FRAMES    = 72
REID_APP_W      = 0.65
REID_IOU_W      = 0.35
REID_MIN_SCORE  = 0.42

# ─────────────────────────────────────────────────────────────────────────────
#  Life-status tags
# ─────────────────────────────────────────────────────────────────────────────
TAGS = [
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
        Creates a new entry if no match found.
        """
        vec = self._compute_vec(crop_bgr)
        if vec is None:
            return None

        # Search existing entries for a close match
        best_key, best_sim = None, MATCH_THRESHOLD - 1e-9
        for stored_key, stored_vec in self._vecs.items():
            sim = self._cosine(vec, stored_vec)
            if sim > best_sim:
                best_sim, best_key = sim, stored_key

        if best_key is not None:
            # Found — update the running average vector for drift robustness
            avg = self._vecs[best_key] * 0.9 + vec * 0.1
            avg /= (np.linalg.norm(avg) + 1e-9)
            self._vecs[best_key] = avg
            self._map[best_key]["seen"] = self._map[best_key].get("seen", 1) + 1
            return self._map[best_key]["quote"], self._map[best_key]["palette_idx"]

        # New person — assign quote + palette
        new_key = self._vec_to_key(vec)
        palette_idx = random.randrange(len(PALETTES))
        self._map[new_key]  = {
            "quote":       random.choice(TAGS),
            "palette_idx": palette_idx,
            "seen":        1,
        }
        self._vecs[new_key] = vec
        self.save()
        return self._map[new_key]["quote"], palette_idx


# ─────────────────────────────────────────────────────────────────────────────
#  Track state + ghost buffer  (per-video)
# ─────────────────────────────────────────────────────────────────────────────
class TrackState:
    __slots__ = ("tag", "palette", "last_box", "last_frame", "appearance")

    def __init__(self, tag: str, palette_idx: int):
        self.tag        = tag
        self.palette    = PALETTES[palette_idx]
        self.last_box   = None
        self.last_frame = 0
        self.appearance = None


_registry: dict[int, TrackState] = {}
_ghosts:   dict[int, TrackState] = {}


def _reset_state():
    _registry.clear()
    _ghosts.clear()


def _get_state(track_id: int) -> TrackState:
    return _registry[track_id]


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
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

    if track_id not in _registry:
        app   = _appearance_hist(frame, x1, y1, x2, y2)
        ghost = _match_ghost(box, app, frame_idx)

        if ghost is not None:
            # Resurrect state from ghost buffer
            _registry[track_id] = _ghosts.pop(ghost)
        else:
            # Ask the hashmap for a persistent quote
            result = hashmap.get_quote(crop) if crop.size > 0 else None
            if result:
                tag, pal = result
            else:
                tag, pal = random.choice(TAGS), random.randrange(len(PALETTES))
            _registry[track_id] = TrackState(tag, pal)

    state = _registry[track_id]
    state.last_box   = box
    state.last_frame = frame_idx
    state.appearance = _appearance_hist(frame, x1, y1, x2, y2)


def _on_track_lost(track_id: int):
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
    if not detections:
        return frame_bgr

    h, w  = frame_bgr.shape[:2]
    fsize = max(20, int(h / 38))
    font  = _font(fsize)

    base    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for track_id, x1, y1, x2, y2 in detections:
        if track_id not in _registry:
            continue
        state           = _get_state(track_id)
        bg_rgba, tx_rgb = state.palette

        tb     = font.getbbox(state.tag)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]

        cx      = (x1 + x2) // 2
        pill_w  = tw + 2 * PAD_X
        pill_h  = th + 2 * PAD_Y
        pill_x0 = int(max(4, min(cx - pill_w // 2, w - pill_w - 4)))
        pill_y0 = int(max(4, y1 - pill_h - 20))
        pill_x1 = pill_x0 + pill_w
        pill_y1 = pill_y0 + pill_h

        draw.line([(cx, y1), (cx, pill_y1)], fill=LINE_C, width=2)
        r = 5
        draw.ellipse([cx-r, y1-r, cx+r, y1+r], fill=(*tx_rgb, 230))
        _rounded_rect(draw, pill_x0, pill_y0, pill_x1, pill_y1, RADIUS, bg_rgba)
        draw.text((pill_x0 + PAD_X, pill_y0 + PAD_Y), state.tag,
                  font=font, fill=(*tx_rgb, 255))
        draw.rectangle([x1, y1, x2, y2], outline=(*tx_rgb, 100), width=2)

    return cv2.cvtColor(
        np.array(Image.alpha_composite(base, overlay).convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  3K scaling + writer
# ─────────────────────────────────────────────────────────────────────────────
def scale_to_3k(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == OUT_WIDTH:
        return frame
    new_h = int(round(h * OUT_WIDTH / w))
    interp = cv2.INTER_LANCZOS4 if w < OUT_WIDTH else cv2.INTER_AREA
    return cv2.resize(frame, (OUT_WIDTH, new_h), interpolation=interp)


def scale_boxes(dets, sw, sh, dw, dh):
    sx, sy = dw / sw, dh / sh
    return [(t, int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
            for t, x1, y1, x2, y2 in dets]


def make_writer(path: Path, w: int, h: int) -> cv2.VideoWriter:
    for fc in ("avc1", "H264", "mp4v"):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fc), OUT_FPS, (w, h))
        if wr.isOpened():
            return wr
    raise RuntimeError("Could not open VideoWriter.")


# ─────────────────────────────────────────────────────────────────────────────
#  Process one video
# ─────────────────────────────────────────────────────────────────────────────
def process_video(input_path: Path, model: YOLO,
                  conf: float, hashmap: PersonHashMap) -> Path | None:
    _reset_state()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open {input_path.name}")
        return None

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_h    = int(round(src_h * OUT_WIDTH / src_w))
    out_path = input_path.parent / f"tagged_{input_path.stem}.mp4"
    writer   = make_writer(out_path, OUT_WIDTH, out_h)

    frame_step = max(1, round(src_fps / OUT_FPS))

    print(f"  {input_path.name}")
    print(f"    Source : {src_w}x{src_h} @ {src_fps:.1f}fps  ({total} frames)")
    print(f"    Output : {OUT_WIDTH}x{out_h} @ {OUT_FPS}fps")

    prev_ids: set[int] = set()
    frame_idx = write_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        results = model.track(
            frame, persist=True, classes=[0],
            conf=conf, verbose=False, tracker="botsort.yaml",
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

        frame_3k   = scale_to_3k(frame)
        oh, ow     = frame_3k.shape[:2]
        dets_sc    = scale_boxes(detections, src_w, src_h, ow, oh)
        tagged     = render_tags(frame_3k, dets_sc)
        writer.write(tagged)

        write_idx += 1
        frame_idx += 1
        if write_idx % 24 == 0:
            pct = frame_idx / total * 100 if total else 0
            print(f"    {write_idx} frames written  ({pct:.0f}%)", end="\r")

    cap.release()
    writer.release()
    hashmap.save()
    print(f"\n  Saved → {out_path.name}  ({write_idx} frames)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Overlay persistent life-status tags on crowd footage."
    )
    parser.add_argument("videos", nargs="*",
                        help="Video files (default: all in ./webcamera/)")
    parser.add_argument("--conf",  type=float, default=0.30)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--print-map", action="store_true",
                        help="Print the full person→quote hashmap and exit")
    args = parser.parse_args()

    hashmap = PersonHashMap()

    if args.print_map:
        hashmap.print_map()
        return

    if args.videos:
        input_files = [Path(v) for v in args.videos]
    else:
        vdir       = Path(__file__).parent / "webcamera"
        exts       = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI"}
        input_files = sorted(f for f in vdir.iterdir()
                             if f.suffix in exts and not f.stem.startswith("tagged_"))

    if not input_files:
        print("No videos found.")
        return

    print(f"Model  : {args.model}")
    print(f"Output : {OUT_WIDTH}px wide · {OUT_FPS}fps")
    print(f"Videos : {len(input_files)}\n")

    model = YOLO(args.model)

    for video in input_files:
        process_video(video, model, conf=args.conf, hashmap=hashmap)

    hashmap.print_map()
    print("All done.")


if __name__ == "__main__":
    main()
