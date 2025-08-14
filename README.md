<div align="center">
<h1><span style="color: rgb(230, 100, 80);">i</span><span style="color: rgb(230, 183, 53);">L</span><span style="color: rgb(117, 160, 85);">R</span><span style="color: rgb(96, 120, 172);">M</span>: An Iterative Large 3D Reconstruction Model</h1>

<a href="https://arxiv.org/abs/2507.23277"><img src="https://img.shields.io/badge/arXiv-2507.23277-b31b1b" alt="arXiv"></a>
<a href="https://gynjn.github.io/iLRM/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

[Gyeongjin Kang](https://gynjn.github.io/info/), [Seungtae Nam](https://github.com/stnamjef), [Xiangyu Sun](https://scholar.google.com/citations?user=VLzxTrAAAAAJ&hl=ko&oi=ao), [Sameh Khamis](https://www.samehkhamis.com), [Abdelrahman Mohamed](https://www.cs.toronto.edu/~asamir/), [Eunbyung Park](https://silverbottlep.github.io/index.html)
</div>

Official repo for the paper "**iLRM: An Iterative Large 3D Reconstruction Model**"

![Teaser Image](/assets//teaser.jpg)

## Installation

```bash
# create conda environment
conda create -n ilrm python=3.10 -y
conda activate ilrm

# install PyTorch (adjust cuda version according to your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## Checkpoints
We first release the 2-view RealEstate10K (256x256) checkpoint, which is the most common baseline in related works and provides a standard reference point for comparison. We will upload other checkpoints soon!

In training and evaluation, we used the dataset preprocessed by [pixelSplat](https://github.com/dcharatan/pixelsplat).


The model checkpoints are host on [HuggingFace](https://huggingface.co/Gynjn/iLRM/tree/main).

| Model | PSNR  | SSIM  | LPIPS |
| ----- | ----- | ----- | ----- |
| [re10k_2view](https://huggingface.co/Gynjn/iLRM/resolve/main/re10k_2view.ckpt?download=true) | 28.49 | 0.899 | 0.113 |

The evaluation results differ from the numbers reported in the paper, mainly because of the data processing precision (using float32 for camera pose calculation). We are currently revising the paper based on our internal discussions, and the updated results will be reflected in the new version. Thank you for your patience and understanding.


## Inference

Update the `dataset.roots` field in `config/experiment/re10k.yaml` with your dataset path.

Update the `checkpointing.load` field in `config/main.yaml` with the pretrained model.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_re10k.json
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