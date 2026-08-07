# Number Gesture Dataset Card

## Dataset summary

- Task: static hand-sign digit classification for `0`–`9`
- Source: Debabrata Kuiry's Kaggle Hand Sign Gesture Dataset (A–Z & 0–9)
- Dataset release: version 1, MIT license, 25,000+ images across 36 classes
- Used subset: numeric class folders only
- Split policy: deterministic, class-balanced 80/10/10 image split
- Storage: local `data/external/`; excluded from Git

## Labels

The model uses the poses shown in the source dataset's `0`–`9` folders. Those
images must be inspected after download; the labels must not be reinterpreted as
the older custom two-hand counting specification.

## Required reporting before release

- Per-label source and split counts
- Corrupt-image validation result
- Lighting and background diversity notes
- Local-webcam holdout performance
- Known collection and pose gaps

## Limitations

The Kaggle description does not expose participant or recording-session IDs.
Consequently, an image-level test split may contain correlated frames and can
overestimate performance on unseen people. Local webcam evaluation must remain
separate from fine-tuning images.
