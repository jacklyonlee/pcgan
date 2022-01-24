# SetVAE
This repository re-implements and reproduces the results of [SetVAE](https://github.com/jw9730/setvae) (CVPR'21).

## Installation
* Set up and activate conda environment.

```shell
conda env create -f environment.yml
conda activate setvae
```

* Compile CUDA extensions.

```shell
sh scripts/install.sh
```

* Download ShapeNet dataset and trained checkpoints.

```shell
sh scripts/download.sh
```

## Training
You can train SetVAE using `train.py` or provided scripts.

```shell
# Train SetVAE using CLI
python train.py --name NAME --cate airplane
# Train SetVAE using provided settings
sh scripts/train_shapenet_aiplane.sh
sh scripts/train_shapenet_car.sh
sh scripts/train_shapenet_chair.sh
```

## Testing
You can evaluate checkpointed models using `test.py` or provided scripts.

```shell
# Test user specified checkpoint using CLI
python test.py --ckpt_path CKPT_PATH --cate car
# Test provided SetVAE checkpoints
sh scripts/test_shapenet_aiplane.sh
sh scripts/test_shapenet_car.sh
sh scripts/test_shapenet_chair.sh
```

## Metrics
Table below shows metrics computed from trained models using the official SetVAE implementation and our implementation. Best scores are highlighed in bold. **MMD-CD** is scaled by 10<sup>3</sup> and **MMD-EMD**, **COV**, **1-NNA** by 10<sup>2</sup>.

| Category  | Model | MMD(↓) CD | MMD(↓) EMD | COV(↑) CD | COV(↑) EMD | 1-NNA(↓) CD | 1-NNA(↓) EMD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Airplane | Official | **0.199** | **3.07** | 43.45 | **44.93** | **75.31** | **77.65** |
|  | Ours     | 0.208 | 3.15 | **48.89** | 43.70 | 77.04 | 81.98 |
| Chair | Official | **2.55** | **7.82** | 46.98 | **45.01** | 58.76 | **61.48** |
|  | Ours     | **2.55** | 7.91 | **47.13** | 44.56 | **58.08** | 62.84 |
| Car | Official | **0.88** | **5.05** | **48.58** | 44.60 | **59.66** | **63.35** |
|  | Ours     | 0.92 | 5.07 | 47.16 | **45.45** | 60.23 | 64.77 |
