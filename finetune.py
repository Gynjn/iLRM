import importlib
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from setup import init_config
from metric_utils import export_results, summarize_evaluation
import tqdm

config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.inference.get("num_threads", 1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
torch.backends.cudnn.allow_tf32 = config.inference.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# Load data
dataset_name = config.inference.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

dataloader = DataLoader(
    dataset,
    batch_size=config.inference.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.inference.num_workers,
    prefetch_factor=config.inference.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
)
dataloader_iter = iter(dataloader)

# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
ILRM = importlib.import_module(module).__dict__[class_name]
model = ILRM(config).to(device)
model.load_ckpt(config.inference.get("ckpt_path", None))

print(f"Running inference; save results to: {config.inference.out_dir}")
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

ft_iter = config.inference.get("finetune_iter", 0)

model.eval()
cnt = 0
for batch in dataloader:
    batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in batch.items()}
    input_data_dict = {key: value[:, :config.data.num_input_frames] if type(value) == torch.Tensor else value for key, value in batch.items()}
    target_data_dict = {key: value[:, config.data.num_input_frames:] if type(value) == torch.Tensor else None for key, value in batch.items()}
    with torch.no_grad(), torch.autocast(
        enabled=config.inference.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.inference.amp_dtype],
    ):
        result = model(input_data_dict, target_data_dict, finetune=True)

    i_fxfycxcy = input_data_dict["fxfycxcy"]
    i_c2w = input_data_dict["c2w"]
    i_image = input_data_dict["image"]
    _, _, _, h, w = i_image.shape
    t_fxfycxcy = target_data_dict["fxfycxcy"]
    t_c2w = target_data_dict["c2w"]

    ### Gaussian parameters optimizer and scheduler
    xyz = nn.Parameter(result.gaussians["xyz"].requires_grad_(True))
    feature = nn.Parameter(result.gaussians["feature"].requires_grad_(True))
    scale = nn.Parameter(result.gaussians["scale"].requires_grad_(True))
    rotation = nn.Parameter(result.gaussians["rotation"].requires_grad_(True))
    opacity = nn.Parameter(result.gaussians["opacity"].requires_grad_(True))  

    l = [
        {'params': xyz, 'lr': 6e-4},
        {'params': feature, 'lr': 1e-3},
        {'params': scale, 'lr': 1e-3},
        {'params': rotation, 'lr': 1e-3},
        {'params': opacity, 'lr': 1e-3}
    ]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=ft_iter)

    for iter in tqdm.tqdm(range(ft_iter), desc=f"Finetuning {ft_iter} iters"):
        optimizer.zero_grad()
        renderings = model.render(
            xyz, feature, scale, rotation, opacity, 
            i_c2w, i_fxfycxcy, w, h
        )
        renderings = renderings.permute(0, 1, 4, 2, 3).contiguous()
        l2_loss = nn.functional.mse_loss(renderings, i_image)        

        loss = l2_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    renderings_target = model.render(
        xyz, feature, scale, rotation, opacity,
        t_c2w, t_fxfycxcy, w, h
    )
    renderings_target = renderings_target.permute(0, 1, 4, 2, 3).contiguous()
    result['render'] = renderings_target
    export_results(result, config.inference.out_dir, 
                    compute_metrics=config.inference.get("compute_metrics"), 
                    save_images=config.inference.get("save_images"),
                    uid=cnt)
    del optimizer, scheduler, xyz, feature, scale, rotation, opacity
    torch.cuda.empty_cache()    
    cnt += 1
torch.cuda.empty_cache()


if config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference.out_dir)
exit(0)
