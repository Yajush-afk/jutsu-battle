# Number Identifier Model Card

## Model details

- Initial architecture: four-block MiniNumberCNN trained from scratch
- Input: `160×160` RGB image
- Output classes: `0`–`9`
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
- The public test split must remain unused during training and model selection.
- Every automated acceptance gate must pass.
- Live webcam throughput must be measured on the target machine.
