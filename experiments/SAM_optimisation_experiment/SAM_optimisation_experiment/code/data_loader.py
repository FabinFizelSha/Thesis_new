#!/usr/bin/env python3
"""
Data Loader for Phase 1 Dataset
Loads RGB-D frames and ground truth semantic masks.
"""

import numpy as np
from pathlib import Path
from typing import Iterator, Dict, Tuple
import os


class Phase1DatasetLoader:
    """
    Loads Phase 1 Suite 1 dataset (300 RGB-D frames with ground truth).

    Expected directory structure:
        dataset_root/
        ├── rgb_XXXXXX.npy          (1280×720×3, uint8)
        ├── depth_XXXXXX.npy        (1280×720, float32, meters)
        └── semantic_XXXXXX.npy     (1280×720 or 1280×720×3, uint8)
    """

    def __init__(self, dataset_root: str, random_shuffle: bool = False):
        """
        Args:
            dataset_root: Path to dataset directory
            random_shuffle: Whether to shuffle frame order
        """
        self.dataset_root = Path(dataset_root)
        self.random_shuffle = random_shuffle

        # Discover all frames
        self.frame_ids = self._discover_frames()

        if random_shuffle:
            import random
            random.shuffle(self.frame_ids)

    def _discover_frames(self) -> list:
        """Find all frame IDs in dataset."""
        frame_ids = set()

        for file in os.listdir(self.dataset_root):
            if file.startswith(('rgb_', 'depth_', 'semantic_')) and file.endswith('.npy'):
                # Extract frame ID from filename (e.g., "rgb_000000.npy" -> "000000")
                frame_id = file.split('_')[1].split('.')[0]
                frame_ids.add(frame_id)

        return sorted(list(frame_ids))

    def __len__(self) -> int:
        """Number of frames in dataset."""
        return len(self.frame_ids)

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over frames."""
        for frame_id in self.frame_ids:
            yield self._load_frame(frame_id)

    def _load_frame(self, frame_id: str) -> Dict:
        """
        Load a single frame.

        Returns:
            Dict with keys:
            - 'rgb': [H, W, 3] uint8
            - 'depth': [H, W] float32 (meters)
            - 'semantic': [H, W] or [H, W, 3] uint8
            - 'frame_id': str
        """
        rgb_path = self.dataset_root / f"rgb_{frame_id}.npy"
        depth_path = self.dataset_root / f"depth_{frame_id}.npy"
        semantic_path = self.dataset_root / f"semantic_{frame_id}.npy"

        rgb = np.load(rgb_path).astype(np.uint8)
        depth = np.load(depth_path).astype(np.float32)
        semantic = np.load(semantic_path).astype(np.uint8)

        # Validate shapes
        assert rgb.shape == (1280, 720, 3), f"RGB shape mismatch: {rgb.shape}"
        assert depth.shape == (1280, 720), f"Depth shape mismatch: {depth.shape}"

        # Semantic can be [H, W] or [H, W, 3]
        if semantic.ndim == 3:
            # If 3-channel, average to single channel
            semantic = semantic.mean(axis=2).astype(np.uint8)

        return {
            'rgb': rgb,
            'depth': depth,
            'semantic': semantic,
            'frame_id': frame_id
        }

    def get_frame(self, index: int) -> Dict:
        """Get frame by index."""
        if index < 0 or index >= len(self.frame_ids):
            raise IndexError(f"Frame index {index} out of range [0, {len(self.frame_ids)})")

        frame_id = self.frame_ids[index]
        return self._load_frame(frame_id)
