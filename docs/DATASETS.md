# Datasets (verify licences yourself before use)
| Dataset | Use | Notes |
|---|---|---|
| GTSDB | Detection benchmark (about 900 images, 1360x800) | Small; use the 4 standard categories (prohibitory, danger, mandatory, other). Adapter: `training/adapters/gtsdb.py`. Check the category mapping against its README. |
| GTSRB | Classification only | Cropped signs. Never a detection claim. |
| TT100K | Large detection set | Chinese signs; check licence (non-commercial restrictions are likely). |
| Indian signs | Deployment | No dataset has been verified for you. Check Roboflow Universe, Mapillary and similar, read the licence and class list, then spot-check 100 labels. Add your own photos (you own the copyright) to fill gaps. Only claim classes you trained on. |
Split by sequence/location (`--group-regex`), never by random image, to avoid leakage. Augmentation is training-only (Ultralytics default, `fliplr=0` because left/right signs differ in meaning).

## Status
Used so far: GTSDB (900 images, 4 groups) for the trained model. Not yet used: GTSRB (classification only), TT100K, any Indian dataset. See INDIAN_SIGNS_PLAN.md.
