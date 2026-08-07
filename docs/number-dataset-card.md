# Number Gesture Dataset Card

## Dataset summary

- Task: canonical two-hand number classification for 0–10 plus `unknown`
- Collection source: local webcam recordings with participant consent
- Participants: pending production collection; target 15 or more
- Sessions: target two per participant
- Split policy: participant-exclusive 10/2/3 train/validation/test split
- Raw-data publication: prohibited by default

## Labels

See `docs/number-gesture-spec.md`. `no_hands` is a detector state and is not a
CNN class.

## Required reporting before release

- Participant count and split membership counts
- Per-label sample and clip counts
- Left/right arrangement coverage for 6–10
- Lighting and background diversity notes
- Known collection gaps
- Consent and deletion process
- Duplicate-rejection settings

## Limitations

Generated smoke fixtures are not part of the production dataset. Accuracy from
random frame-level splits is invalid because nearby frames from one participant
are correlated.
