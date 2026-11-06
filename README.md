COMP9501 Paper Reading — Reproduction Code

Paper: Lixin Duan, Dong Xu, Ivor Wai-Hung Tsang, and Jiebo Luo,
"Visual Event Recognition in Videos by Learning from Web Data,"
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2012.

This repository reproduces five methods on real frames with SIFT, dense HOF, and HSV color:
SVM_T, SVM_ST (average of 24 kernels), MKL without MMD, A-MKL, and A-MKL without pre-learned classifiers.

DATA_ROOT/
  source/{animal,water,city,vehicle}/*.mp4
  target/{animal,water,city,vehicle}/*.mp4

Use 30 source clips and 24 target clips per class. Each of 10 repeats labels 3 target clips per class for training and tests the rest.

```
pip install -r requirements.txt
DATA_ROOT=/path/to/data python run.py
```

Feature cache: ./cache/features.npz
