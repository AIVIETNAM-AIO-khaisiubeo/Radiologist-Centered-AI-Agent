"""Feedback log: every accept / edit / reject (and deferred manual read) is an event.

Correctness is NOT stored as "what the reader said". Events keep the reader's
decisions; whether a decision was right is judged later against the reference
annotation, so a reader's mistakes never silently become ground truth.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FeedbackEvent:
    reader_id: str
    case_id: str
    policy: str
    shown: bool
    ai_label: int | None
    ai_probability: float
    ai_confidence: float | None
    uncertainty: float
    threshold: float
    initial_read: int | None
    action: str                       # accept | edit | reject | manual | defer
    final_label: int | None           # None when the reader deferred the decision
    source: str = "simulated"         # simulated | live
    edit: dict | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    # audit context: what exactly was shown when the decision was made
    session_id: str | None = None
    policy_version: str | None = None
    model_version: str | None = None
    recommendation: dict | None = None  # label text, probability, threshold, uncertainty, tool agreement
    impression_at: str | None = None
    revealed_at: str | None = None
    supersedes: str | None = None      # timestamp of the decision this one revises (never overwritten)

    def to_dict(self) -> dict:
        return asdict(self)


class FeedbackStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, reader_id: str) -> Path:
        return self.root / f"{reader_id}.jsonl"

    def append(self, ev: FeedbackEvent | dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        d = ev.to_dict() if isinstance(ev, FeedbackEvent) else ev
        with open(self.path(d["reader_id"]), "a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")

    def events(self, reader_id: str) -> list[dict]:
        p = self.path(reader_id)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def latest_by_case(self, reader_id: str) -> dict[str, dict]:
        out = {}
        for e in self.events(reader_id):
            out[e["case_id"]] = e
        return out

    def reset(self, reader_id: str) -> None:
        p = self.path(reader_id)
        if p.exists():
            p.unlink()
