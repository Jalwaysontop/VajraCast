# VajraCast

**Thunderstorm and lightning nowcasting, from observation frames to district-level warnings.**

Built for Smart India Hackathon 2026, problem statement 26072 (IMD). VajraCast takes radar, satellite, lightning and NWP feeds on a common 2 km grid and 10-minute clock, forecasts the next two hours, and serves the results to a web console that shows storm cells, district warnings and model skill.

---

## What it does

| Output | Description |
|---|---|
| **Reflectivity nowcast** | Column-max reflectivity (dBZ), 10 to 120 min ahead in 10-min steps, on a 2 km grid |
| **Lightning probability** | P(at least 1 flash within 10 km) in the next 30 and 60 min, isotonic-calibrated |
| **First-flash flag** | High probability where there was no lightning in the past 30 min |
| **Degraded mode** | One checkpoint serves both full inputs (radar + satellite + lightning + NWP) and satellite-only inputs, trained with radar modality dropout |
| **Ensemble members** (optional) | A diffusion residual refiner samples sharp, reproducible members around the deterministic forecast |
| **Baselines** | Persistence, Lucas-Kanade extrapolation and a 20-member pysteps STEPS ensemble, all with the same output schema |
| **Cells, districts, warnings** | The backend turns forecasts into tracked cells, district-level summaries and warnings for the console |

## Architecture

```
observation frames -> backend (rolling buffer -> nowcast_ml model -> cells, districts, warnings)
                          -> /api/v1/console -> web console

```

The repository has three parts:

- **`ml/`** is the `nowcast_ml` package. It holds only the machine-learning part: training, calibration, evaluation, export and inference. It has no web framework and exposes one class, `Predictor`.
- **`backend/`** is the `nowcast_backend` service. It keeps a rolling buffer of observation frames, calls the `nowcast_ml` model, and produces cells, districts and warnings. It serves the console API under `/api/v1/console`.
- **`web/`** is the console, with views for the map, storm cells, warnings, model skill and system status.

## Run the whole application

```
make install     # ml + backend into ml/.venv, npm install in web/
make demo        # backend on :8000 and console on http://localhost:5173
```

`make demo` runs on dummy data. The first time, the backend trains a small model on synthetic storms (about 90 s). It then replays a synthetic afternoon over Odisha: 19 forecast times, with radar lost for the last hour to show the satellite-only fallback. Storms, lightning, warnings and the skill scores in the console are all generated.

`make test` runs the ML and backend test suites and type-checks the console.

## Moving to real data

1. Produce input-contract stores. The format is specified in [`ml/docs/data_contract.md`](ml/docs/data_contract.md).
2. Train with the stages described in [`ml/README.md`](ml/README.md).
3. Point `NOWCAST_MODEL_PATH` at the resulting model artifact.
4. Push frames to the backend, as described in [`backend/README.md`](backend/README.md).

The console needs no change.

## Repository layout

```
.
├── backend/                    nowcast_backend service
│   ├── src/nowcast_backend/
│   │   ├── adapters/
│   │   ├── api/
│   │   │   ├── console/        console API
│   │   │   └── routes/
│   │   ├── domain/
│   │   └── services/           rolling buffer, nowcasting, cells, districts, warnings
│   └── tests/
├── ml/                         nowcast_ml package
│   ├── configs/                YAML configs: data, eval, model, train
│   ├── docs/                   data contract
│   ├── notebooks/              Colab training notebook
│   ├── scripts/                synthetic events, SEVIR download and conversion, pipelines
│   ├── src/nowcast_ml/
│   │   ├── baselines/          persistence, extrapolation, STEPS
│   │   ├── calibration/        isotonic calibration
│   │   ├── data/
│   │   ├── evaluation/
│   │   ├── inference/          Predictor, schema, errors
│   │   ├── models/
│   │   │   └── refiner/        diffusion residual refiner
│   │   ├── training/
│   │   └── utils/
│   └── tests/
│       ├── integration/
│       └── unit/
└── web/                        web console
    ├── public/
    └── src/
        ├── app/
        ├── assets/geo/
        ├── components/
        ├── data/
        │   └── api/            calls to the backend
        ├── features/
        │   ├── cells/
        │   ├── map/
        │   ├── skill/
        │   ├── status/
        │   └── warnings/
        ├── lib/
        ├── store/
        └── styles/
```

## The ML package

### Training stages

Each stage is one command with its own config. Full details and options are in [`ml/README.md`](ml/README.md).

| Stage | What it does |
|---|---|
| (a) Pretrain | SEVIR data (VIL to max-z approximation, IR, GLM), 128x128, 7 to 12 frames at 10 min |
| (b) Fine-tune | India events; backbone frozen for a set number of epochs, then unfrozen at a lower learning rate |
| (c) Lightning head | Only the lightning head trains, with focal loss |
| (d) Calibration | An isotonic curve per (mode, lead) is written into the artifact |
| (e) Ensemble (optional) | Diffusion residual refiner trained with the base model frozen |

Dataset splits are always **by event** and are recorded in each artifact's `splits.json`.

### Using the model from code

Load once at startup, then call `predict` for each new analysis time. `Predictor` holds no mutable state after `load` and imports no web framework.

```python
import xarray as xr
from nowcast_ml.inference import InputContractError, ModelLoadError, Predictor

predictor = Predictor.load("artifacts/models/nowcast/latest", device="auto")

# At least 7 frames (10-min, UTC) in the input-contract format; missing channels are allowed.
inputs = xr.open_zarr("data/events/synth_00000.zarr").isel(time=slice(0, 12))
t0 = inputs["time"].values[-1]

forecast = predictor.predict(inputs, t0)
refl_60 = forecast["reflectivity"].sel(lead=60)   # dBZ, (y, x)
p_ltg_30 = forecast["lightning_prob_30"]          # calibrated probability, (y, x)
new_cells = forecast["first_flash"]               # bool, (y, x)
```

### Output contract

| Variable | Dims | Meaning |
|---|---|---|
| `reflectivity` | `(lead, y, x)` | Forecast max-z in dBZ, `lead` = 10 to 120 min |
| `lightning_prob_30`, `lightning_prob_60` | `(y, x)` | Calibrated P(at least 1 flash within 10 km) in the next 30 / 60 min |
| `first_flash` | `(y, x)` | `lightning_prob_30` above threshold and no flash within 10 km in the past 30 min |
| `reflectivity_members` | `(member, lead, y, x)` | Optional ensemble: the STEPS baseline, or the model with the refiner |

Attributes include `model_name`, `model_version`, `t0`, `mode` (`full` or `satellite_only`), `missing_channels` and `inference_ms`.

### Model artifacts

Each trained model is a versioned folder with weights, config, channel list, normalization stats, calibrator, splits, metrics and a model card. `manifest.json` holds a SHA-256 for every file, and loading fails with `ModelLoadError` if a file is missing, modified, or inconsistent with the others. `nowcast-export` writes TorchScript and ONNX versions and checks both against eager PyTorch.

### Measured inference speed

Full-size model (10.1 M parameters), one forward pass, random weights:

| Device | 256x256 (512 km) | 512x512 (1024 km) |
|---|---|---|
| CPU, Apple M4 Pro | 607 ms | 1735 ms |
| MPS, Apple M4 Pro | 76 ms | 284 ms |
| Colab T4 | not measured yet | |

Ensemble members cost extra. The refiner took 1.6 s per member on CPU and 0.37 s per member on MPS at 256x256, so use a GPU for ensembles. The deterministic path is unaffected.

## Testing

`make test` runs the ML and backend test suites and type-checks the console. The ML suite has more than 125 offline CPU tests on synthetic data, covering the schema validator, hand-computed metric values, baseline motion direction, a one-batch overfit check, artifact round-trip and tamper detection, TorchScript/ONNX parity, `Predictor` behaviour in both modes, and CLI smoke tests.

## Known limits

- No real-data skill numbers exist yet. The SEVIR loader was tested against a fake file with the published SEVIR layout; one test checks the orientation assumptions once real files are downloaded.
- SEVIR VIL-to-dBZ is an approximate inversion. India fine-tuning is expected to correct the intensity.
- The deterministic model blurs with lead time. Refiner members are sharper, but whether their spread is well calibrated can only be judged on real data.
- STEPS at convective scales (about 5 to 10 km cells) loses skill quickly on small domains, which is the expected behaviour of its noise model.

## Further reading

- [`ml/README.md`](ml/README.md): training, evaluation, export, CLI reference
- [`ml/docs/data_contract.md`](ml/docs/data_contract.md): input format for the data team
- [`backend/README.md`](backend/README.md): pushing frames to the backend
