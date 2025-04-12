import os

import torch
import torch.nn as nn
import math
from typing import Union, Optional, Literal
from safetensors.torch import save_file

class LoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Module, rank: int, multiplier: float, alpha: float,weight_dtype: torch.bfloat16):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.multiplier = multiplier
        self.alpha = alpha
        self.org_forward = None
        self.weight_dtype = weight_dtype

        if isinstance(base_layer, nn.Linear):
            # 线性层的 LoRA 低秩矩阵
            self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
            self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)

            # 添加新的调整层
            self.adjust_layer = nn.Linear(1, self.lora_B.out_features, bias=False)
            nn.init.ones_(self.adjust_layer.weight)

        elif isinstance(base_layer, nn.Conv2d):
            # LoRA 低秩卷积
            self.lora_A = nn.Conv2d(
                base_layer.in_channels, rank, base_layer.kernel_size, base_layer.stride, base_layer.padding, bias=False
            )
            self.lora_B = nn.Conv2d(
                rank, base_layer.out_channels, (1, 1), (1, 1), padding=0, bias=False
            )
            # 添加新的调整层
            # 添加新的调整层
            self.adjust_layer = nn.Linear(1, self.lora_B.out_channels, bias=False)
            nn.init.ones_(self.adjust_layer.weight)

        else:
            raise ValueError("Unsupported base layer type. Must be nn.Linear or nn.Conv2d")

        # 初始化 LoRA 参数
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(1))
        self.lora_A.weight.data = self.lora_A.weight.data.to(weight_dtype)

        nn.init.zeros_(self.lora_B.weight)
        self.lora_B.weight.data = self.lora_B.weight.data.to(weight_dtype)

        self.adjust_layer.weight.data = self.adjust_layer.weight.data.to(weight_dtype)

    def apply_to(self):
        """ 替换原始 forward 方法 """
        self.org_forward = self.base_layer.forward  # 备份原始 forward
        self.base_layer.forward = self.forward  # 替换 forward 方法
        del self.base_layer

    def forward(self, x):

        lora_output = self.lora_B(self.lora_A(x))
        scale_factor = torch.tensor([self.multiplier]).to(x.device).to(self.weight_dtype)
        scale_factor = self.adjust_layer(scale_factor)
        if x.dim() == 4:
            scale_factor = scale_factor.view(1, -1, 1, 1)
        elif x.dim() == 2:
            scale_factor = scale_factor.view(1, -1)
        elif x.dim() == 3:  # 修正: 这里要用 `elif` 而不是 `else`
            scale_factor = scale_factor.view(1, 1, -1)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        lora_output = lora_output * scale_factor


        # return self.org_forward(x)
        return self.org_forward(x) + lora_output * (self.alpha / self.rank)
        #
        # return self.org_forward(x) + lora_output


def create_modules(
        prefix: str,
        root_module: nn.Module,
        rank: int,
        multiplier: float,
        alpha: float = 1.0,
        weight_dtype = torch.bfloat16
) -> list:
    loras = []
    names = []
    for name, module in root_module.named_modules():
        if "attn2" in name or "time_embed" in name:
            continue
        if module.__class__.__name__ in [
            "ResnetBlock2D",
            "Downsample2D",
            "Upsample2D",
            "DownBlock2D",
            "UpBlock2D",
            "Attention"
        ]:
            for child_name, child_module in module.named_modules():
                if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear",
                                                       "LoRACompatibleConv"]:
                    lora_name = f"{prefix}.{name}.{child_name}".replace(".", "_")
                    lora = LoRALayer(child_module, rank, multiplier, alpha, weight_dtype=torch.bfloat16)
                    lora.lora_name = lora_name

                    if lora_name not in names:
                        loras.append(lora)
                        names.append(lora_name)

    # for name, module in root_module.named_modules():
    #     # 仅对 ResnetBlock2D 的 Conv2d 和 Transformer2DModel 的 Linear 进行微调
    #     if "ResnetBlock2D" in module.__class__.__name__:
    #         for child_name, child_module in module.named_modules():
    #             if isinstance(child_module, (nn.Conv2d, nn.Linear)):  # 处理 Conv2d
    #                 lora_name = f"{prefix}.{name}.{child_name}".replace(".", "_")
    #                 lora = LoRALayer(child_module, rank, multiplier, alpha, weight_dtype=torch.bfloat16)
    #                 lora.lora_name = lora_name
    #
    #                 if lora_name not in names:
    #                     loras.append(lora)
    #                     names.append(lora_name)
    #
    #     elif "Transformer2DModel" in module.__class__.__name__:
    #         for child_name, child_module in module.named_modules():
    #             if isinstance(child_module, nn.Linear):  # 处理 Linear 但仅限于 attn1 和 attn2
    #                 if "attn1" in child_name :
    #                     lora_name = f"{prefix}.{name}.{child_name}".replace(".", "_")
    #                     lora = LoRALayer(child_module, rank, multiplier, alpha,weight_dtype=torch.bfloat16)
    #                     lora.lora_name = lora_name
    #
    #                     if lora_name not in names:
    #                         loras.append(lora)
    #                         names.append(lora_name)
    #     elif any(layer in module.__class__.__name__ for layer in
    #              ["Downsample2D", "Upsample2D", "DownBlock2D", "UpBlock2D"]):
    #         for child_name, child_module in module.named_modules():
    #             if isinstance(child_module, nn.Conv2d):  # 仅对 Down/Up Blocks 的 Conv2d 添加 LoRA
    #                 lora_name = f"{prefix}.{name}.{child_name}".replace(".", "_")
    #                 lora = LoRALayer(child_module, rank, multiplier, alpha, weight_dtype=torch.bfloat16)
    #                 lora.lora_name = lora_name
    #
    #                 if lora_name not in names:
    #                     loras.append(lora)
    #                     names.append(lora_name)

    return loras


class LoRAModule(nn.Module):
    def __init__(self, root_module: nn.Module, rank: int = 4, multiplier: float = 0.1, alpha: float = 1.0,weight_dtype: torch.bfloat16 = torch.bfloat16):
        super().__init__()

        self.lora_scale = None
        self.lora_modules = create_modules(
            prefix="unet",
            root_module=root_module,
            rank=rank,
            multiplier=multiplier,
            alpha=alpha,
            weight_dtype = weight_dtype
        )
        print(f"create LoRA for U-Net: {len(self.lora_modules)} modules.")

        lora_names = set()
        for lora in self.lora_modules:
            assert (
                lora.lora_name not in lora_names
            ), f"Duplicated LoRA name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)

        # 应用 LoRA 适配器
        for lora in self.lora_modules:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        del root_module
        torch.cuda.empty_cache()

    def prepare_optimizer_params(self, train_model: Optional[Literal["LoRAModule", "AdjustModule"]] = None):
        all_params = []

        if train_model == "LoRAModule":
            # 仅更新 LoRA 相关参数 (lora_A 和 lora_B)
            params = [param for lora in self.lora_modules for param in
                      (lora.lora_A.parameters(), lora.lora_B.parameters())]
            all_params.append({"params": [p for param_group in params for p in param_group]})  # 展开嵌套列表

        elif train_model == "AdjustModule":
            # 仅更新 AdjustLayer（强化学习优化）
            params = [param for lora in self.lora_modules for param in lora.adjust_layer.parameters()]
            all_params.append({"params": params})

        return all_params

    def perturb_lora_params(self, noise_std: float = 0.1, scale_factor: float = 1.0):
        """
        1. 提取 LoRAModule 中的所有 LoRA 相关参数
        2. 对参数施加扰动（加噪声或缩放）
        3. 计算图能够追踪变化
        4. 评估扰动是否合理
        """
        params = self.prepare_optimizer_params(train_model="AdjustModule")[0]["params"]

        param_trajectories = []  # 存储参数轨迹 (原始参数, 修改后参数)

        for param in params:
            if param.requires_grad:
                original_param = param.clone()  # 备份原始参数

                # 施加扰动
                perturbation = noise_std * torch.randn_like(param)

                # 计算扰动统计信息
                # print(f"Perturbation mean: {perturbation.mean().item()}, std: {perturbation.std().item()}")

                # 限制 scale_factor，防止极端变化
                scale_factor = torch.empty(1).uniform_(0.9, 1.1).item()

                # 添加小 epsilon 避免梯度爆炸
                epsilon = 1e-1
                param = param * scale_factor + perturbation * epsilon  # 让 PyTorch 追踪计算图

                # 存储 (原始参数, 变化后的参数)
                param_trajectories.append((original_param, param.clone()))

        return param_trajectories  # 返回参数的修改轨迹

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        #         for key in list(state_dict.keys()):
        #             if not key.startswith("lora"):
        #                 # lora以外除外
        #                 del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.lora_modules:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.lora_modules:
            lora.multiplier = 0


if __name__ == "__main__":
    from diffusers import AudioLDMPipeline,DDIMScheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_id = "/data/HDD1/tjut_makunsheng/Music-mm/huggingface/audioldm"
    pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    unet = pipe.unet
    unet.eval()
    vae = pipe.vae
    vae.eval()
    scheduler = pipe.scheduler

    lora_handler = LoRAModule(root_module=unet, rank=4, multiplier=0.1, alpha=1.0)


    # 打印模型
    print("yes")