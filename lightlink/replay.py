"""Run the full receiver over any frame source (simulator, saved video) without hardware."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .receiver import Receiver


@dataclass
class RunResult:
    timeline: list = field(default_factory=list)   # (t, state, quality)
    events: list = field(default_factory=list)     # FrameEvent in decode order
    frames: int = 0
    offsets: list = field(default_factory=list)    # (t, dx, dy)

    def first_time(self, state, after=0.0):
        for t, s, _ in self.timeline:
            if t >= after and s == state:
                return t
        return None


def run_frames(cfg, frames, on_update=None) -> RunResult:
    rx = Receiver(cfg)
    res = RunResult()
    for f in frames:
        t, img = (f.t, f.image) if hasattr(f, "image") else f
        u = rx.process(t, img)
        res.frames += 1
        res.timeline.append((t, u.state, u.quality))
        res.events += u.events
        if u.offset is not None:
            res.offsets.append((t, *u.offset))
        if on_update:
            on_update(u, img)
    return res


def save_video(path, frames, fps, fourcc="MJPG"):
    """Write frames to `path` plus `path.ts.csv` with true timestamps (video files lose them)."""
    path = Path(path)
    writer, stamps = None, []
    for f in frames:
        img = cv2.cvtColor(f.image, cv2.COLOR_GRAY2BGR)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        writer.write(img)
        stamps.append(f.t)
    if writer:
        writer.release()
    path.with_suffix(path.suffix + ".ts.csv").write_text("\n".join(f"{t:.6f}" for t in stamps))


def read_video(path):
    """Yield (t, bgr). Uses the .ts.csv sidecar if present, else frame index / fps."""
    path = Path(path)
    side = path.with_suffix(path.suffix + ".ts.csv")
    stamps = [float(s) for s in side.read_text().split()] if side.exists() else None
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield (stamps[i] if stamps and i < len(stamps) else i / fps), img
        i += 1
    cap.release()
