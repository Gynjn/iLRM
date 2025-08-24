<div align="center">
<h1><span style="color: rgb(230, 100, 80);">i</span><span style="color: rgb(230, 183, 53);">L</span><span style="color: rgb(117, 160, 85);">R</span><span style="color: rgb(96, 120, 172);">M</span>: An Iterative Large 3D Reconstruction Model</h1>

<a href="https://arxiv.org/abs/2507.23277"><img src="https://img.shields.io/badge/arXiv-2507.23277-b31b1b" alt="arXiv"></a>
<a href="https://gynjn.github.io/iLRM/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

[Gyeongjin Kang](https://gynjn.github.io/info/), [Seungtae Nam](https://github.com/stnamjef), [Xiangyu Sun](https://scholar.google.com/citations?user=VLzxTrAAAAAJ&hl=ko&oi=ao), [Sameh Khamis](https://www.samehkhamis.com), [Abdelrahman Mohamed](https://www.cs.toronto.edu/~asamir/), [Eunbyung Park](https://silverbottlep.github.io/index.html)
</div>

Official repo for the paper "**iLRM: An Iterative Large 3D Reconstruction Model**"

This branch contains the code for the high-resolution (960x540) undistorted DL3DV dataset.

## Installation

```bash
# create conda environment
conda create -n ilrm python=3.11 -y
conda activate ilrm

# install PyTorch (adjust cuda version according to your system)
pip install -r requirements.txt
```

## Checkpoints
The model checkpoints are host on [HuggingFace](https://huggingface.co/Gynjn/iLRM/tree/main).

| Model | PSNR  | SSIM  | LPIPS |
| ----- | ----- | ----- | ----- |
| [undistored_dl3dv](https://huggingface.co/Gynjn/iLRM/resolve/main/ilrm_undistort_dl3dv.pt?download=true) | 24.25 | 0.802 | 0.258 |

This checkpoint was trained with 32 input images. We recommend finetuning it when using a different number of input images.

For training and evaluation, we used the DL3DV dataset after applying undistortion preprocessing with this [script](https://github.com/arthurhero/Long-LRM/blob/main/data/prosess_dl3dv.py), originally introduced in [Long-LRM](https://arthurhero.github.io/projects/llrm/index.html). 

Download the DL3DV benchmark dataset from [here](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/tree/main), and apply undistortion preprocessing.

## Inference

Update the `inference.ckpt_path` field in `configs/ilrm.yaml` with the pretrained model.

Update the entries in `data/dl3dv_eval.txt` to point to the correct processed dataset path.

You can save videos or images by changing the fields (`inference.save_video` or `save_images`) in `configs/ilrm.yaml`.

The number of finetuning (post-prediction optimization) iterations is set to 10 in `inference.finetune_iter`.

```bash
# inference
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/ilrm.yaml

# post-prediction optimization
CUDA_VISIBLE_DEVICES=0 python finetine.py --config configs/ilrm.yaml
```

## Citation

```
@article{kang2025ilrm,
  title={iLRM: An Iterative Large 3D Reconstruction Model},
  author={Kang, Gyeongjin and Nam, Seungtae and Sun, Xiangyu and Khamis, Sameh and Mohamed, Abdelrahman and Park, Eunbyung},
  journal={arXiv preprint arXiv:2507.23277},
  year={2025}
}
```

## Acknowledgements

This branch is built on many amazing research works, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [gsplat](https://github.com/nerfstudio-project/gsplat)
- [LVSM](https://github.com/haian-jin/LVSM)
- [Long-LRM](https://github.com/arthurhero/Long-LRM)
- [LaCT](https://github.com/a1600012888/LaCT)