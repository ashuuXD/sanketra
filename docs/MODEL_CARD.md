# Model card: sanketra_gtsdb_n

## Summary
- Architecture: Ultralytics YOLO11n (2.6 M parameters, 5.5 MB), fine-tuned from COCO-pretrained weights.
- Task: detect traffic signs in a road image and assign one of four **groups**: prohibitory, danger, mandatory, other.
- Training data: GTSDB (German Traffic Sign Detection Benchmark, FullIJCNN2013, 900 images, 1,213 boxes, 159 images without signs kept as negatives). **German signs. Not Indian signs.**
- Split: by blocks of 10 consecutive image numbers (90 groups), seed 0: 631 train / 149 val / 120 test images.
- Training: 30 epochs, batch 8, image size 640, AdamW (auto), horizontal flip disabled (left/right signs differ), 8.1 minutes on an RTX 3050 Laptop 4 GB. One run, seed 0. Ultralytics 8.4.182, PyTorch 2.11 + CUDA 12.8.
- Companion model (optional, in the app): generic yolo11n for people and vehicles, never used as a sign detector.

## Measured results (this machine, single run; see experiments/results in your SanketraData folder)
Held-out **test** split (120 images, 160 boxes), image size 640:
| | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| all | 0.937 | 0.879 | 0.929 | 0.739 |
| prohibitory | 0.975 | 0.988 | 0.985 | 0.845 |
| danger | 0.953 | 0.950 | 0.956 | 0.758 |
| mandatory | 0.877 | 0.714 | 0.842 | 0.631 |
| other | 0.944 | 0.865 | 0.933 | 0.723 |
Validation split (used to choose the best epoch, so optimistic): mAP@0.5 0.951, mAP@0.5:0.95 0.793.

Speed (GPU, median of 50 runs, 720p input): 15.0 ms at size 320, 16.1 ms at 416, 15.7 ms at 640 (about 62 to 66 FPS for the model alone).

Robustness, mAP@0.5 on the test split under **simulated** corruptions, severity 0 (clean) to 5:
| corruption | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| brightness (overexposure) | 0.929 | 0.930 | 0.929 | 0.902 | 0.864 | 0.838 |
| blur | 0.929 | 0.926 | 0.913 | 0.895 | 0.851 | 0.806 |
| noise | 0.929 | 0.909 | 0.862 | 0.762 | 0.679 | 0.578 |
| low resolution | 0.929 | 0.927 | 0.899 | 0.845 | 0.777 | 0.683 |
| low light | 0.929 | 0.905 | 0.804 | 0.651 | 0.538 | 0.447 |
Precision stays high while recall falls (low light, severity 5: precision 0.761, recall 0.403): under hard conditions the model mostly **misses** signs instead of inventing them.

Calibration: fitted temperature 0.52 (on validation); expected calibration error on test 0.122 before, 0.070 after.

## Intended use
Education, demonstrations, and as a baseline for studying robustness and uncertainty in sign perception.

## Not intended / limitations
- Not a safety system; the app says so. Never use it for driving decisions.
- Group-level only: it cannot tell a 50 km/h limit from a no-overtaking sign.
- Trained and evaluated on one small German dataset (test: 160 boxes). This 30-epoch model is a single run; the 100-epoch yolo11n has three training seeds and yolo11s one (see below). Intervals cover training randomness only, not test-set sampling error.
- Does not recognise Indian signs. Domain shift (signs, colours, clutter, lighting) is expected to cost accuracy.
- Corruptions are simulated and are not real night or rain footage.
- Known false positives on non-sign blue/red objects in street scenes.
- Licence note: Ultralytics is AGPL-3.0; GTSDB has its own terms of use. Check both before sharing models or data.

## Second model: sanketra_gtsdb_s (yolo11s) and comparison
yolo11s (9.4 M parameters, 19.2 MB), same data and split, **100 epochs (34 minutes)**, batch 8, 640 px, seed 0.
The first yolo11n above had 30 epochs and yolo11s 100, so that pair was confounded. The controlled comparison (yolo11n retrained for 100 epochs, same data, split, seed and recipe) is below.

Test split, image size 640:
| | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | median ms | size |
|---|---|---|---|---|---|---|
| yolo11n, 30 epochs | 0.937 | 0.879 | 0.929 | 0.739 | 15.7* | 5.5 MB |
| yolo11n, 100 epochs (seed 0) | 0.920 | 0.879 | 0.929 | 0.767 | 13.6* | 5.5 MB |
| yolo11s, 100 epochs | 0.974 | 0.902 | 0.970 | 0.847 | 19.0* | 19.2 MB |
*single latency measurements, 640 px; they are noisy (see "Latency is noisy" below) and do not support a speed comparison.
Per class, yolo11s (P / R / mAP@0.5 / mAP@0.5:0.95): prohibitory 1.000 / 0.967 / 0.993 / 0.903; danger 0.984 / 1.000 / 0.995 / 0.887; mandatory 0.910 / 0.762 / 0.909 / 0.771; other 1.000 / 0.881 / 0.984 / 0.828. Mandatory is still the weakest group.

Robustness of yolo11s (mAP@0.5, severity 0 to 5): brightness 0.968, 0.962, 0.956, 0.940, 0.913, 0.887; blur 0.968, 0.966, 0.952, 0.919, 0.869, 0.797; noise 0.968, 0.957, 0.901, 0.789, 0.681, 0.575; low resolution 0.968, 0.964, 0.936, 0.879, 0.785, 0.673; low light 0.968, 0.947, 0.885, 0.690, 0.572, 0.448.
At the worst severities the two models converge (low light severity 5: 0.448 for yolo11s versus 0.447 for yolo11n; noise 0.575 versus 0.578), while yolo11s is better at mild and medium severities. Recall at low light severity 5 is 0.381 (yolo11s) versus 0.403 (yolo11n), so extra capacity did not fix the failure mode (silent misses). Calibration: both fitted temperature 0.52; ECE 0.109 -> 0.044 (yolo11s) versus 0.122 -> 0.070 (yolo11n). One run each.

## Controlled comparison: yolo11n with 100 epochs (sanketra_gtsdb_n100)
yolo11n, same data, split, seed 0, batch 8, 640 px and recipe as yolo11s, **100 epochs, 22.7 minutes** (0.379 h) on the RTX 3050 Laptop. Ultralytics 8.4.194. One run.
Test split (120 images, 160 boxes), image size 640:
| | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| all | 0.920 | 0.879 | 0.929 | 0.767 |
| prohibitory | 0.975 | 0.988 | 0.991 | 0.870 |
| danger | 0.877 | 0.950 | 0.979 | 0.814 |
| mandatory | 0.887 | 0.748 | 0.859 | 0.702 |
| other | 0.942 | 0.830 | 0.889 | 0.684 |
Validation split (used to choose the best epoch, so optimistic): mAP@0.5 0.969, mAP@0.5:0.95 0.830.

What the three runs show (all on the same 120 test images):
- **More epochs for yolo11n (30 -> 100):** mAP@0.5 unchanged (0.929), mAP@0.5:0.95 0.739 -> 0.767 (+0.028): tighter boxes, not more signs found.
- **Larger model at equal training (yolo11n 100 -> yolo11s 100):** mAP@0.5 0.929 -> 0.970 (+0.041; +0.035 against the three-seed mean 0.935), mAP@0.5:0.95 0.767 -> 0.847 (+0.080). Of the 0.108 mAP@0.5:0.95 gap seen in the first comparison, about 0.028 was training length and about 0.080 model size.
- Mandatory remains the weakest group for both sizes (recall 0.748 for yolo11n, 0.762 for yolo11s).
- Caveat: see the three-seed result below. 160 test boxes remain a small test set, and no bootstrap over test images was done.

Speed (GPU, median of 50 runs, 720p input): 14.9 ms at size 320, 14.7 ms at 416, 13.6 ms at 640 on 31 Oct. Repeat measurements of the same architecture varied much more on other days: see "Latency is noisy" at the end; no speed difference to yolo11s is claimed.

Robustness of yolo11n 100 epochs (mAP@0.5 on test, severity 0 to 5; severity 0 is 0.931 because corrupted copies go through the same evaluation path, versus 0.929 in the clean report: a 0.001 difference whose cause was not traced): brightness 0.931, 0.931, 0.915, 0.885, 0.857, 0.824; blur 0.931, 0.921, 0.911, 0.873, 0.842, 0.817; noise 0.931, 0.916, 0.859, 0.779, 0.686, 0.575; low resolution 0.931, 0.914, 0.884, 0.842, 0.785, 0.661; low light 0.931, 0.902, 0.824, 0.676, 0.562, 0.446.
At low light severity 5: precision 0.842, recall 0.389 (misses again dominate). At severity 5 all three models land in the same place (yolo11n 100 epochs / yolo11n 30 epochs / yolo11s; low light 0.446 / 0.447 / 0.448, noise 0.575 / 0.578 / 0.575); yolo11s is clearly better only at mild and medium severities (for example low light severity 2: 0.885 versus 0.824).

Calibration (temperature fitted on validation, ECE on test): yolo11n 100 epochs T = 0.56, ECE 0.088 -> 0.045; yolo11n 30 epochs T = 0.52, ECE 0.122 -> 0.070; yolo11s T = 0.52, ECE 0.109 -> 0.044. Temperatures below 1 mean raw scores were under-confident in all three.

## Informal observations on Indian street video (31 Oct 2026; not measured, no labels, no accuracy claim)
Played INDRA clip video95 (Anand, Gujarat, head-mounted camera) in the Console with the yolo11n 100-epoch sign model plus the generic COCO model.
- A **Domino's Pizza shop sign** (red/blue/white round logo) was repeatedly detected as an "Other" sign (several track IDs, seen in the screenshot). Most of these boxes were drawn dashed amber, i.e. flagged uncertain by the app's abstention rule; I did not count how many were not.
- Auto-rickshaws were labelled Truck or Bus by the generic COCO model (it has no auto-rickshaw class).
- Some other "Other sign" detections near the right edge of the frame may be vehicle lights: **not verified**.
- Conclusion: the German-trained model produces false alarms on Indian street clutter (domain shift, as expected). This is an observation, not a measured precision; measuring it needs a manually reviewed sample of events.
- Temporal study (3 short clips, no labels): too few tracks to report as a result; see the README/report wording.

## Calibration in the live app
The live temperature slider defaults to 1.0 (configs/default.yaml), i.e. **no calibration is applied by default**. The fitted values (0.52 to 0.56) were estimated on German validation data and used only in the offline evaluation (ECE). They are deliberately not applied to Indian footage: a temperature below 1 raises scores, and nothing shows it transfers to a different domain.

## Temporal smoothing study (31 Oct 2026; no ground truth, stability only)
Question: does smoothing the confidence of a tracked object over time reduce display flicker? `evaluation.temporal_eval` runs detector + ByteTrack over a video and, for every track seen in at least 10 frames, compares three display rules at threshold 0.5: **raw** (show when this frame's confidence >= 0.5), **EMA** (alpha 0.5 on the track's confidence) and **EMA + hold 3** (also keep the box through up to 3 missed frames). Flicker rate = share of consecutive frames in which the shown/hidden state flips (lower = steadier). Settings were the defaults, not tuned.
Footage: four unlabelled clips on Indian roads (video95 and video99 from the INDRA dataset, whose Kaggle licence is listed as unknown; a Kanpur highway clip; and video01.mp4, a 360p dashcam-style clip, first 3000 frames). The sources and licences of the Kanpur clip and video01 were not recorded; the clips were used privately and are not redistributed. Models: the generic COCO yolo11n (people and vehicles) and, separately, the sign model.

**Generic COCO yolo11n (people, vehicles), 290 tracks in total:**
| clip | tracks | raw | EMA | EMA + hold 3 |
|---|---|---|---|---|
| Kanpur highway | 14 | 0.121 | 0.048 | 0.048 |
| video01 (3000 frames) | 192 | 0.155 | 0.082 | 0.078 |
| video95 | 50 | 0.063 | 0.034 | 0.033 |
| video99 | 34 | 0.073 | 0.036 | 0.034 |
| pooled (track-weighted) | 290 | 0.128 | 0.067 | 0.064 |
EMA lowered flicker by 46 to 60 percent in every clip (pooled about 48 percent); hold 3 added a further 0 to 4 percentage points of reduction (pooled about 50 percent). Both clips with the most tracks agree in direction and size.

**Sign model (yolo11n 100 epochs), video01, 6000 frames, only 14 tracks:** raw 0.234, EMA 0.120, EMA + hold 3 0.098. Earlier sign-model runs on the short clips gave 0 to 8 tracks each and are not a sample. The German-trained sign model finds few stable sign tracks on this footage.

Caveats (important):
- This measures **stability, not accuracy**. A stable wrong detection (for example the Domino's sign) scores well; smoothing can also delay a real detection or keep showing a vanished one. Latency and missed-detection cost of smoothing were **not** measured.
- Tracks are not independent (video01 supplies 192 of 290), the clips are few and unlabelled, and there is one run with default settings. Do not quote the percentages as precise.
- The headline result is on a generic COCO model; the sign model's own sample is too small to conclude anything about signs.

## Run-to-run variation (measured 5 Nov 2026)
yolo11n, 100 epochs, same data and split, training seeds 0, 1, 2 (`python -m evaluation.seed_summary --tags gtsdb_n100 gtsdb_n100_s1 gtsdb_n100_s2`):
| metric (test) | mean | min / max | sd | 95% t-interval (n=3) |
|---|---|---|---|---|
| mAP@0.5 | 0.935 | 0.929 / 0.942 | 0.006 | 0.919 to 0.951 |
| mAP@0.5:0.95 | 0.767 | 0.765 / 0.767 | 0.001 | 0.764 to 0.769 |
yolo11s (one run, 0.970 / 0.847) lies outside both intervals, so the size gap is larger than yolo11n's training-seed spread. Not covered: test-set sampling error (120 images, 160 boxes), seed variation of yolo11s, and the software version (seed 0 ran on Ultralytics 8.4.194, seeds 1 and 2 on 8.4.197). The 3-sample interval is wide and approximate.

## Latency is noisy on this laptop (do not quote one number)
Model-only median latency (benchmark_latency, 50 runs, GPU) for the same yolo11n architecture: 14.9 / 14.7 / 13.6 ms at 320 / 416 / 640 px on 31 Oct (seed 0), 16.0 / 19.4 / 27.5 ms on 5 Nov (seed 2; p95 25.8 / 35.7 / 95.1 ms), and 15.7 ms at 640 px for the 30-epoch run; yolo11s 19.0 ms at 640 px. The yolo11n range overlaps yolo11s, so **no speed difference between the two sizes is claimed**. The GPU sat in its idle power state (P8, 210 MHz) before the 5 Nov run, and power mode, background load and the antivirus were not controlled.
Whole-pipeline numbers read by eye from screen recordings of the Console (detector + second model + tracker + drawing, with screen recording running; indicative only): about 9 to 11 FPS and 32 to 49 ms per frame on 2 Nov; 1.8 to 7.6 FPS and 94 to 500 ms on 4 Nov (a training run was probably using the GPU at the same time, not confirmed); 7 to 14 FPS and 55 to 513 ms on 5 Nov with yolo11s. Frame times spike, so treat the live demo as "several frames per second on a 4 GB laptop GPU", not as a measured real-time guarantee.
