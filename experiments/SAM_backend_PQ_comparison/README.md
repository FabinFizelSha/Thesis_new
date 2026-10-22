# SAM backend comparison (NanoSAM vs SAM ViT-B), class-agnostic PQ

Stand-alone benchmark on 500 uHumans2 office frames. Method: `results/method.md`. Metric: `src/pq_eval.py` (Kirillov et al., CVPR 2019).

Setup (Jetson AGX Thor): venv with system site packages that can see torch/TensorRT and the editable `segment_anything`, `nanosam`, `torch2trt`
(`~/.venvs/sam_pq`). Models are not in git: SAM checkpoint `~/rsg_models/sam/sam_vit_b_01ec64.pth`, NanoSAM ONNX in `~/third_party/nanosam/data/`;
build the engines into `models/nanosam/` with the `trtexec` commands in `results/method.md`/NanoSAM README (engines are GPU/TensorRT-specific).

    python src/prep_data.py <phase1_frames_300 dir>   # data/frames, data/gt (frames 300-499 need src/extract_extra.py first)
    python src/sanity.py                              # metric sanity checks
    python src/run_A.py                               # Experiment A: both models at 3x3/5x5/7x7 -> results/comparison.csv
