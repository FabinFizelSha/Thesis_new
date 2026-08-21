"""Phase 1 Dataset Loader for frame iteration"""

import numpy as np
from pathlib import Path


class Phase1DatasetLoader:
    """Load Phase 1 dataset frames from numpy files"""

    def __init__(self, dataset_dir, random_shuffle=False):
        self.dataset_dir = Path(dataset_dir)
        self.random_shuffle = random_shuffle

        # Find all frame indices
        rgb_files = sorted(self.dataset_dir.glob('rgb_*.npy'))
        self.frame_indices = [int(f.stem.split('_')[1]) for f in rgb_files]

        if self.random_shuffle:
            np.random.shuffle(self.frame_indices)

    def __iter__(self):
        for frame_idx in self.frame_indices:
            rgb_file = self.dataset_dir / f'rgb_{frame_idx:06d}.npy'
            depth_file = self.dataset_dir / f'depth_{frame_idx:06d}.npy'
            semantic_file = self.dataset_dir / f'semantic_{frame_idx:06d}.npy'

            if rgb_file.exists() and depth_file.exists() and semantic_file.exists():
                rgb = np.load(rgb_file)
                depth = np.load(depth_file)
                semantic = np.load(semantic_file)

                yield {
                    'frame_id': frame_idx,
                    'rgb': rgb,
                    'depth': depth,
                    'semantic': semantic
                }
