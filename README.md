# Point Cloud GAN
This repository implements a modified version of [Point Cloud GAN](https://github.com/chunliangli/Point-Cloud-GAN) (ICLR'19 Workshop) which achieves performance comparable to the [SetVAE](https://github.com/jw9730/setvae) in point cloud generation.

## Installation
* Set up and activate conda environment.

```shell
conda env create -f environment.yml
conda activate pcgan
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
You can train using `train.py` or provided scripts.

```shell
# Train using CLI
python train.py --name NAME --cate airplane
# Train using provided settings
sh scripts/train_shapenet_aiplane.sh
```

## Testing
You can evaluate checkpointed models using `test.py` or provided scripts.

```shell
# Test user specified checkpoint using CLI
python test.py --ckpt_path CKPT_PATH --cate car
# Test provided checkpoints
sh scripts/test_shapenet_aiplane.sh
```

## Metrics
Table below shows final metrics for SetVAE and our model (**MMD-CD** is scaled by 10<sup>3</sup> and **MMD-EMD**, **COV**, **1-NNA** by 10<sup>2</sup>). SetVAE is trained for 8000 epochs and our model is trained for 2000 epochs.

| Category  | Model | MMD(↓) CD | MMD(↓) EMD | COV(↑) CD | COV(↑) EMD |
| :---: | :---: | :---: | :---: | :---: | :---: | 
| Airplane | SetVAE | 0.199 | 3.07 | 43.45 | 44.93 |
|  | Ours     | 0.224 | 3.45 | 38.27 | 36.79 |

