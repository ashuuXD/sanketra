# Architecture
```mermaid
flowchart LR
  S[Webcam / video file] --> C[OpenCV capture]
  C --> D[Sign model YOLO11n + ByteTrack]
  C --> O[Optional objects model: people, vehicles]
  D --> K[Temperature calibration + EMA smoothing]
  K --> A[Abstain flag]
  A --> R[Sign registry]
  O --> M[Merge]
  R --> M
  M --> V[Overlay + event log]
  V --> W[Binary WebSocket: meta JSON + JPEG]
  W --> U[React console, canvas painting]
  V --> L[(Bounded event log)]
  L --> X[CSV / JSON export]
  H[(SanketraData: models, datasets, results)] -.-> D
  H -.-> E[evaluation scripts] -.-> P[Results page]
```
Components: backend (API, engine, events, registry, settings), vision (corruptions, calibration, temporal, drawing, paths), training (dataset preparation, GTSDB adapter, training), evaluation (detection report, latency, calibration, robustness, temporal), frontend (Home, Console, Perception Lab, Results, Signs).
Detection finds boxes; classification labels a crop (here the detector also classifies the group); tracking keeps identities; interpretation maps a class to registry meaning. The sign model is the only source of sign claims.
Data flow rule: code lives in the project folder (replaceable); your models, datasets, results and uploads live in SanketraData (persistent).
