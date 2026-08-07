# Number Identifier Model Card

## Model details

- Selected architecture: pending validation comparison
- Input: padded `160×160` RGB union crop or normalized landmarks
- Output classes: `0`–`10`, `unknown`
- Hand detector state: `no_hands`
- Runtime: local PyTorch inference

## Release metrics

The following fields must be copied from the guarded final evaluation rather
than entered from memory:

```text
Unseen-user macro-F1:
Minimum per-class recall:
Unknown false-acceptance rate:
Expected calibration error:
Median inference latency:
p95 inference latency:
Model size:
```

## Intended use

Real-time recognition of the canonical finger-number poses documented in the
gesture specification. It is not a general sign-language translator and does
not accept arbitrary finger combinations with the same total count.

## Release restrictions

- Smoke checkpoints cannot be released.
- The test participants must remain unseen during training and model selection.
- Every automated acceptance gate must pass.
- Live webcam throughput must be measured on the target machine.
