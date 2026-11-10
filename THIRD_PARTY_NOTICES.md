# Third-party notices

This repository contains the SANKETRA source code and documentation only. It does not contain datasets, trained weights or videos.

| Component | Use here | Licence (check the project's own LICENSE file before redistributing) |
|---|---|---|
| Ultralytics (YOLO11, ByteTrack implementation) | detection, training, tracking | AGPL-3.0 (Ultralytics also offers a separate enterprise licence) |
| PyTorch / torchvision | inference and training | BSD-style |
| OpenCV (opencv-python) | video, image processing | Apache-2.0 |
| FastAPI, Uvicorn | backend | MIT / BSD |
| React, Vite, TypeScript | web interface | MIT / MIT / Apache-2.0 |
| `data/samples/street_bus.jpg`, `tests/assets/street.jpg` | demo and test image: the standard `bus.jpg` sample image of the Ultralytics project (810x1080) | AGPL-3.0, as part of the Ultralytics project |

Because the project builds on Ultralytics (AGPL-3.0), this repository is released under the AGPL-3.0 as well (see `LICENSE`). This is a project decision, not legal advice.

## Data and models not included
- **GTSDB** (German Traffic Sign Detection Benchmark): obtain it yourself from https://benchmark.ini.rub.de/gtsdb_dataset.html and follow its terms. It is not redistributed here, and weights trained on it are not published here.
- **COCO-pretrained yolo11n weights**: downloaded automatically by Ultralytics on first use; not stored in this repository.
- **Video clips** used for informal tests (INDRA dataset clips and other road videos) are not included and not redistributed. The INDRA dataset (Brahmbhatt, arXiv:2211.07916) lists its licence as unknown on its Kaggle page.
