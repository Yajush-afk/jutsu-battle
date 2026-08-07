# Two-Hand Number Gesture Specification

## Purpose

This document is the source of truth for the Phase 1 number-recognition task.
The recognizer identifies canonical finger-count poses for the integers 0
through 10 and rejects visible but unsupported poses as `unknown`.

## Label vocabulary

The CNN has twelve output labels in this exact order:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, unknown
```

`no_hands` is an application state produced by the hand detector. It is not a
CNN output because there is no hand crop to classify.

## Canonical poses

Hands should face generally toward the camera, remain inside the capture guide,
and have enough separation for the extended fingers to be visible.

| Label | Required hands | Canonical pose |
|---:|---:|---|
| 0 | 1 | One closed fist |
| 1 | 1 | Index finger extended |
| 2 | 1 | Index and middle fingers extended |
| 3 | 1 | Index, middle, and ring fingers extended |
| 4 | 1 | Four non-thumb fingers extended |
| 5 | 1 | All five fingers extended |
| 6 | 2 | One open hand plus one index finger |
| 7 | 2 | One open hand plus index and middle fingers |
| 8 | 2 | One open hand plus index, middle, and ring fingers |
| 9 | 2 | One open hand plus four non-thumb fingers |
| 10 | 2 | Two open hands |

For labels 6 through 10, either physical hand may show five. Mirrored camera
views, left-handed users, and swapped hand positions do not change the label.

## Unknown and invalid poses

Label a visible sample `unknown` when any of the following applies:

- The finger combination is not canonical for a supported number.
- The required number of hands is wrong.
- A hand is severely cropped or important fingers are hidden.
- The user is moving between poses.
- The hands overlap enough to make the pose ambiguous.
- A hand is edge-on, facing away, or otherwise unreadable.
- An unrelated hand pose or gesture is shown.

Do not label a failed hand detection as `unknown`; that is the `no_hands`
application state. A closed fist detected successfully is label 0.

## Collection protocol

- Recruit 15 participants and record two sessions per participant.
- Record at least four clips for every numeric label in every session.
- Record at least ten `unknown` or transition clips per session.
- Vary lighting, background, distance, hand ordering, and small rotations.
- Capture both left- and right-side arrangements for labels 6 through 10.
- Store participant identity as an anonymous ID such as `subject_001`.
- Obtain consent before recording and never commit raw recordings to Git.
- Sample at most five diverse frames from one clip for a training manifest.

## Locked evaluation split

Splits are assigned by participant, never by image or video frame:

```text
10 participants: training
 2 participants: validation
 3 participants: final test
```

The test participants and acceptance thresholds must be locked before model
selection. Test results are produced only after the final model is selected.

## Acceptance targets

```text
Unseen-user macro-F1:      at least 0.90
Minimum per-class recall:  at least 0.80
Unknown false acceptance:  at most 0.10
End-to-end camera rate:    at least 20 FPS
Stable output latency:     at most 750 ms
```
