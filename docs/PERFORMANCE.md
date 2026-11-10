# Performance notes (measured on a laptop with an RTX 3050 Laptop GPU (4 GB), torch 2.11 + cu128)
Output of `python scripts\bench_gpu.py` (yolo11n, ms; "first" = first call at a new input shape):

| setting | imgsz | frame | first | median | p95 |
|---|---|---|---|---|---|
| rect fp16 | 640 | 960x540 | 7023 (also CUDA init) | 15.6 | 24.7 |
| rect fp16 | 640 | 640x480 | 2857 | 18.0 | 21.0 |
| square fp16 | 640 | 960x540 | 3074 | 16.0 | 38.1 |
| square fp16 | 640 | 640x480 | 29 (shape already seen) | 15.9 | 21.5 |
| rect fp32 | 640 | 960x540 | 397 | 16.3 | 22.2 |
| rect fp32 | 640 | 640x480 | 105 | 14.8 | 20.2 |
| square fp32 | 640 | 960x540 | 236 | 15.5 | 17.8 |
| square fp32 | 640 | 640x480 | 16 | 14.7 | 15.8 |
| rect fp32 | 416 | 960x540 | 195 | 15.1 | 17.1 |
| square fp32 | 416 | 640x480 | 20 | 15.3 | 18.3 |
Draw + JPEG encode of a 960x540 frame: 2.3 ms.

Findings: FP16 costs about 3 s the first time each input shape is seen and gives no steady-state speed-up on this GPU, so the app uses FP32 (`model.fp16: false`). Median inference is about 15 ms for 320 to 640, so the GPU is not the bottleneck at these sizes.
