COMP9501 Paper Reading — Reproduction Code

Paper: Lixin Duan, Dong Xu, Ivor Wai-Hung Tsang, and Jiebo Luo,
"Visual Event Recognition in Videos by Learning from Web Data,"
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2012.

Two stages:

1. Data. `prepare.py` reads captions from `RAW_ROOT/source` and `RAW_ROOT/target`, labels animal / water / city / vehicle, and writes the clips.
2. Result. `run.py` builds 24 ASTPM kernels (SIFT, dense HOF, HSV color) and evaluates SVM_S, SVM_T, SVM_ST, FR, MKL, A-MKL, and A-MKL_noPre. `results/metrics.json` stores accuracy, macro precision, and per-class precision over 10 test draws.

Each clip is one `.mp4` and a `.txt` caption with the same stem. Put them in tar shards or directly in the domain folder:

```
RAW_ROOT/source/*.tar
RAW_ROOT/target/*.tar
```

```
RAW_ROOT=/path/to/raw DATA_ROOT=data python prepare.py
DATA_ROOT=data python run.py
```

`RAW_ROOT=/path/to/raw python run.py` runs both stages. Feature cache: `cache/features.npz`.

Selection: keyword in the first 140 characters, class order animal, water, city, vehicle. City words are skyline, downtown, skyscraper, cityscape, metropolis. Vehicle words must fall in the first 80 characters. At most one clip per shard. Animal and vehicle keywords are capped at 3 clips. Near-duplicate first sentences are removed (ratio 0.66). Quotas: 30 source and 24 target clips per class. Each of 10 repeats (`numpy` seeds 1000–1009) labels 3 target clips per class and tests the rest.
