# Jutsu Battle

Jutsu Battle is a zero-cost, offline-first computer-vision game in which a
player performs hand signs through a webcam to cast jutsu against a CPU
opponent. Development begins with a two-hand finger-number classifier and then
reuses that vision pipeline for hand-seal sequences and combat.

## Project status

Phase 0 is complete when the repository, Python package, Conda environment, and
continuous-integration checks are operational. No model or dataset is included
at this stage.

Planned milestones:

1. Two-hand number recognition for values 0 through 10.
2. A reusable real-time gesture event engine.
3. Twelve-class hand-seal recognition and sequence decoding.
4. A timed, player-versus-CPU elemental battle game.
5. Testing, optimization, accessibility, and release packaging.

## Setup

The supported development environment uses Python 3.11.

```bash
conda env create -f environment.yml
conda activate naruto
python -m pip install --editable . --no-deps
```

The default Linux dependency set can use a supported NVIDIA GPU. Contributors
who need a smaller CPU-only environment can install the matching PyTorch and
TorchVision wheels from the official PyTorch CPU index before installing the
remaining requirements.

Run the checks:

```bash
ruff check .
mypy src
pytest --cov=src
```

## Cost and privacy constraints

- The project does not require paid APIs, hosting, datasets, or assets.
- Webcam processing and inference are designed to run locally.
- Raw participant recordings, private datasets, checkpoints, and captured
  videos are excluded from Git.
- Camera frames will not be saved outside explicit data-collection modes.

## Fan-project notice

This is an unofficial, noncommercial fan project inspired by *Naruto*. It is
not affiliated with, endorsed by, or sponsored by the rights holders. The code
license applies only to original project code; it does not grant rights to
third-party names, datasets, media, or other intellectual property.

## License

Original source code in this repository is available under the MIT License.
Third-party datasets and assets retain their own licenses and attribution
requirements.
