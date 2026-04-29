from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

import pinnstorch


# ==================== 修改区 ====================
def make_read_data_fn(noise_level=0.0):
    """工厂函数：生成 read_data_fn，允许添加可配置的高斯噪声"""
    def read_data_fn(root_path):
        data = pinnstorch.utils.load_data(root_path, "burgers_shock.mat")
        exact_u = np.real(data["usol"])
        if noise_level > 0.0:
            rng = np.random.default_rng(seed=1234)  # 固定种子保证可复现
            noise = noise_level * np.std(exact_u) * rng.normal(size=exact_u.shape)
            exact_u = exact_u + noise
        return {"u": exact_u}
    return read_data_fn

def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):   
    """Define the partial differential equations (PDEs)."""

    u_x, u_t = pinnstorch.utils.gradient(outputs["u"], [x, t])
    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    outputs["f"] = u_t + outputs["u"] * u_x - (0.01 / np.pi) * u_xx

    return outputs

def pde_fn_null(outputs: Dict[str, torch.Tensor],
                x: torch.Tensor,
                t: torch.Tensor):
    """空 PDE 函数：返回零残差，完全移除物理约束。用于实验 C 对照组。"""
    outputs["f"] = torch.zeros_like(outputs["u"])
    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnstorch.utils.extras(cfg)

    # train the model

    # 根据配置选择 PDE 函数
    if cfg.get("use_pde", True):
        selected_pde_fn = pde_fn          # 正常 PINN
    else:
        selected_pde_fn = pde_fn_null     # 纯数据驱动

    # 根据配置生成 read_data_fn，支持噪声
    noise_level = cfg.get("noise_level", 0.0)
    selected_read_fn = make_read_data_fn(noise_level)

    metric_dict, _ = pinnstorch.train(
        cfg, read_data_fn=selected_read_fn, pde_fn=selected_pde_fn, output_fn=None
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
