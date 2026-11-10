# Plan: Indian road-sign support (not done yet)
The current model does NOT recognise Indian signs. Do not claim it does.
1. **Choose classes first.** Pick 10 to 15 signs you can verify from the official source (Motor Vehicles Act 1988 First Schedule / IRC:67). Put exact names, meanings and the source in configs/sign_registry.json, set verified=true only after checking.
2. **Collect data you may legally use.** Your own photos and dashcam clips (you own the copyright) taken in different light, weather, distance and angle. Look at public datasets only after reading each licence and class list; spot-check 100 labels before trusting any set.
3. **Label.** Use a labelling tool (for example Label Studio, CVAT or Roboflow), export YOLO format, and keep one frame per scene group so the split is by location/sequence.
4. **How much.** Aim for at least 100 to 150 labelled instances per class; fewer classes with more examples beats many weak ones. Note the count per class in the report.
5. **Train two ways and compare.** (a) from COCO weights, (b) from the GTSDB model (transfer). Use the same split and report test metrics per class, robustness curves and calibration exactly as for GTSDB.
6. **Write the claim narrowly.** "Detects N verified Indian sign classes on X test images", never "recognises Indian signs".
7. **Check domain shift honestly.** Evaluate the GTSDB model on your Indian test set before fine-tuning and report that number too.

## Candidate public datasets (found by search; I could not open or download them, so licence, access terms and annotation format are UNVERIFIED)
- **IRTSD-Datasetv1** (IEEE DataPort, DOI 10.21227/ap4b-c439): about 5,141 phone photos from many Indian cities, 37 sign classes, PNG, not split into train/test. The page lists images in per-class folders; it is not clear whether bounding boxes are included.
- **ITSDB** (IEEE DataPort, DOI 10.21227/sppg-r994): about 1,264 images in YOLO format with a single class ("traffic sign"): useful for localisation, not for telling signs apart.
IEEE DataPort needs an account and has its own terms; read them and cite the dataset as it asks.

## How to use whatever you download
`.\scripts\phase7_indian.ps1 -Src "C:\path\to\extracted\dataset"` auto-detects the layout:
- images + YOLO .txt labels, or Pascal VOC .xml: converted directly;
- per-class folders without boxes: add `-Proposal "C:\Users\<you>\SanketraData\models\sanketra_gtsdb_s.pt"` so a detector proposes one box per image (labels become machine-made: review a sample, report it);
- single-class YOLO data (ITSDB): train it as a sign localiser; identification needs a classifier or the per-class data.
It then splits, adds registry stubs (meanings to fill in from the official source), trains yolo11s, evaluates, and the Home status flips to "Done" once the model knows at least 5 registry signs.

## Kaggle "Indian Traffic Sign Dataset" (checked on the downloaded zip, 30 Oct 2026): NOT usable as a detector dataset
What the zip actually contains (counted, not assumed): 13,972 files, 59 class folders (`Images/0` ... `Images/58`, folder 39 is missing), 32x32 pixel PNGs (some binary black/white), about 200 files per class (a few classes 600), plus `traffic_sign.csv` (ClassId, Name). The Kaggle page says CC0 and that each class image was taken from a government website, then scaled and augmented.
Problems found:
- It is a cropped-sign classification set, not road scenes: there are no boxes, and a detector would only learn "the whole 32x32 image is the sign".
- The files in a class are augmented copies of very few source pictures (file names chain `..._original_43.png_<id>.png_<id>.png`). A random train/test split would put copies of the same picture in both sets, so any accuracy would be inflated.
- Several class folders do not match their CSV names (for example classes 2 and 3 share the name "One-way traffic"; class 36 is an all-black square; some images are mirrored or upside down). Every label would need manual checking against IRC:67 before use.
Use it at most for: (a) pictogram reference images in the Signs page after you verify them, or (b) a clearly-labelled classifier experiment on cropped signs, reported with the leakage caveat. Do not report it as "Indian road-sign detection".
Importer guard: `indian_import` now refuses folder datasets whose images are mostly smaller than 64 px, and numbered folder names ("0", "1") stay distinct (`class_0`, `class_1`) instead of all becoming "unnamed".
Not suitable either: Kaggle "IRSTD-Data" (IRSTD-1K) is infrared small-target detection, unrelated to road signs.

## Registry meanings (1 Nov 2026)
configs/sign_registry.json v0.2.0: 9 of the 10 Indian entries now carry wording checked against the Delhi Traffic Police public sign guide (traffic.delhipolice.gov.in/en/node/108) and are marked verified with that source and date; "sharp_curve" has no exact official name and stays unverified. The MVA 1988 Schedule shows signs as pictures (its web text has no written meanings) and IRC:67 was not re-read, so "verified" means "wording checked against that guide". Hindi labels remain empty: copy them from the Hindi version of the same page (language switch at the top of the page) into `label_hi`, in `SanketraData\sign_registry.local.json`, after checking them yourself.

## Status (6 Nov 2026): Indian-sign model skipped
No suitable labelled Indian sign dataset was obtained for this project. The one public candidate found (IRTSD-Datasetv1 on IEEE DataPort: CC BY 4.0, 1.78 GB, 5,141 images, 37 classes) states on its page that access requires an IEEE DataPort subscription, so it was not downloaded or tested and is not cited in the report. To continue later: get access (library or institutional subscription, or ask the authors), extract with `tar` to a short path outside Downloads, run `.\scripts\phase7_indian.ps1 -Src <folder>`, split by source (not random frames), and report it as a new, separately evaluated model.
Another candidate seen on 7 Nov 2026 (not used): Roboflow Universe "India Traffic Sign Computer Vision Dataset" (by "college", object detection, CC BY 4.0, 3,751 images in the dataset, version v9 with 10,213 images after augmentation, exports available as YOLOv8, YOLOv11, VOC and others, 245 downloads). Unknown: the class list and balance, where the images came from, and whether near-identical images sit in both train and test. The local download is the "yolov4pytorch" export, which this project's importer does NOT read (it reads YOLO txt per image, VOC xml and class folders); a re-export as YOLOv8 would be needed. The Kaggle 32x32 set re-downloaded the same day is the classification-only set rejected above.
Hindi sign names: no authoritative Hindi source was found (the official Hindi page of the Delhi Traffic Police guide could not be fetched, and the Hindi pages found were commercial). `label_hi` stays empty until they are copied from an official source and checked from an official source.
