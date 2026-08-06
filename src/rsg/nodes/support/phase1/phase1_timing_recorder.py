"""Buffered CSV timing/debug recorder for Phase 1 nodes."""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple


class Phase1TimingRecorder:
    """Buffer Phase 1 timing events and write plain CSV outside the hot path."""

    def __init__(self, enabled: bool, output_path: str, autosave_every: int, logger: Any, sheet_name: str) -> None:
        self.enabled = enabled
        self.output_path = Path(output_path).expanduser().resolve()
        self.autosave_every = max(0, int(autosave_every))
        self.logger = logger
        # Kept in the signature for configuration compatibility. CSV has no sheets.
        self.sheet_name = sheet_name
        self.samples: List[Dict[str, Any]] = []
        self._lock = Lock()
        self._save_thread: Optional[Thread] = None
        self._save_pending = False
        self._last_saved_count = 0

        if self.enabled:
            self.logger.info(f"Phase 1 buffered CSV timing enabled. Output: {self.output_path}")
        else:
            self.logger.info("Phase 1 buffered CSV timing disabled.")

    def add_sample(self, **sample: Any) -> None:
        """Add one event. Accepts arbitrary columns to support both Phase 1 nodes."""
        if not self.enabled:
            return
        with self._lock:
            self.samples.append(dict(sample))
            count = len(self.samples)
        if self.autosave_every > 0 and count % self.autosave_every == 0:
            self.request_save_async()

    def request_save_async(self) -> None:
        """Run the `request save async` operation."""
        if not self.enabled:
            return
        with self._lock:
            if self._save_thread is not None and self._save_thread.is_alive():
                self._save_pending = True
                return
            self._save_pending = False
            self._save_thread = Thread(target=self._save_worker, daemon=True)
            self._save_thread.start()

    def _save_worker(self) -> None:
        while True:
            snapshot = self._snapshot_for_save()
            if snapshot is not None:
                samples, count = snapshot
                self._write_csv(samples, count)
            with self._lock:
                if self._save_pending:
                    self._save_pending = False
                    continue
                return

    def _snapshot_for_save(self) -> Optional[Tuple[List[Dict[str, Any]], int]]:
        with self._lock:
            count = len(self.samples)
            if count == 0:
                return None
            if count == self._last_saved_count and self.output_path.exists():
                return None
            return list(self.samples), count

    def save(self) -> None:
        """Synchronously write latest timing data. Intended for node shutdown."""
        if not self.enabled:
            return
        thread = self._save_thread
        if thread is not None and thread.is_alive():
            thread.join()
        snapshot = self._snapshot_for_save()
        if snapshot is None:
            return
        samples, count = snapshot
        self._write_csv(samples, count)

    def _write_csv(self, samples: List[Dict[str, Any]], count: int) -> None:
        if not self.enabled or not samples:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        headers: List[str] = []
        for sample in samples:
            for key in sample.keys():
                if key not in headers:
                    headers.append(key)

        temporary_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(samples)
        temporary_path.replace(self.output_path)
        self._last_saved_count = count
        self.logger.info(f"Saved Phase 1 timing CSV with {count} rows: {self.output_path}")
