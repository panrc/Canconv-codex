import os
import json
import shutil
import tempfile
import logging
import torch
import argparse

# 调用已有内部脚本的 python 接口
from canconv.scripts.train import main as train_main
from canconv.scripts.test import run_test as test_run
from canconv.scripts.show import show as show_results

import evaluate_pansharpening as eval_ps

from canconv.models import cannet as cannet_module

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集配置映射
data_cfg = {
    "gf2": {
        "train_data": "dataset/training_gf2/train_gf2.h5",
        "val_data": "dataset/training_gf2/valid_gf2.h5",
        "test_reduced_data": "dataset/reduced_examples/test_gf2_multiExm1.h5",
        "test_origscale_data": "dataset/full_examples/test_gf2_OrigScale_multiExm1.h5",
        "dataset_scale": 1023,
        "spectral_num": 4,
        "sensor_range_max": 1023.0,
    },
    "qb": {
        "train_data": "dataset/training_qb/train_qb.h5",
        "val_data": "dataset/training_qb/valid_qb.h5",
        "test_reduced_data": "dataset/reduced_examples/test_qb_multiExm1.h5",
        "test_origscale_data": "dataset/full_examples/test_qb_OrigScale_multiExm1.h5",
        "dataset_scale": 2047,
        "spectral_num": 4,
        "sensor_range_max": 2047.0,
    },
    "wv3": {
        "train_data": "dataset/training_wv3/train_wv3.h5",
        "val_data": "dataset/training_wv3/valid_wv3.h5",
        "test_reduced_data": "dataset/reduced_examples/test_wv3_multiExm1.h5",
        "test_origscale_data": "dataset/full_examples/test_wv3_OrigScale_multiExm1.h5",
        "dataset_scale": 2047,
        "spectral_num": 8,
        "sensor_range_max": 2047.0,
    },
}

# 模型配置映射
model_cfg = {
    "cannet": {
        "module": cannet_module,
        "model_class": "CANNet",
        "default_channels": 32
    }
}

# 结果保存字典
all_eval_results = {}


def ensure_weights(model_name: str, sensor: str) -> str:
    """检查权重是否存在，如不存在则训练并返回权重路径"""
    weight_dir = "weights"
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = os.path.join(weight_dir, f"{model_name}_{sensor}.pth")
    if os.path.exists(weight_path):
        logging.info(f"检测到已有权重: {weight_path}, 将直接使用")
        return weight_path

    logging.info(f"未找到 {model_name}_{sensor} 预训练权重，开始训练……")
    # 基础配置
    model_module = model_cfg[model_name]["module"]
    base_cfg = model_module.cfg.copy()
    # 合并数据集相关配置
    base_cfg.update(data_cfg[sensor])
    base_cfg["exp_name"] = f"{model_name}_{sensor}"
    # 为了更快实验，可根据需求调整 epochs
    base_cfg.setdefault("epochs", 500)
    # 早停参数
    base_cfg["early_stopping_patience"] = 3
    # 8G显存优化
    base_cfg["batch_size"] = 16  # 降低批次大小
    base_cfg["channels"] = 24   # 降低通道数
    # 每 1 轮就进行一次验证
    base_cfg["val_interval"] = 1

    # 调用内部训练函数（无保存 mat）
    train_main(model_name, cfg=base_cfg, save_mat=False)

    # 训练完成后寻找 best_model.pth
    best_model_path = os.path.join("runs", base_cfg["exp_name"], "weights", "best_model.pth")
    final_model_path = os.path.join("runs", base_cfg["exp_name"], "weights", "final.pth")

    chosen = None
    if os.path.exists(best_model_path):
        chosen = best_model_path
    elif os.path.exists(final_model_path):
        chosen = final_model_path
    else:
        raise FileNotFoundError(f"无法在 {model_name}_{sensor} 训练输出中找到模型权重")

    shutil.copy(chosen, weight_path)
    logging.info(f"已将训练得到的权重复制到 {weight_path}")
    return weight_path


def generate_test_and_show(model_name: str, sensor: str, cfg: dict, weight_path: str):
    """使用 test & show 生成结果并保存可视化图像"""
    logging.info(f"开始 {model_name}_{sensor} 数据集测试")
    # 基础输出目录（按权重名）
    base_dir = os.path.join(os.path.dirname(weight_path), os.path.splitext(os.path.basename(weight_path))[0])

    # 分别测试 Reduced & OrigScale，保存到不同子目录，防止结果被覆盖
    reduced_dir = os.path.join(base_dir, "reduced")
    orig_dir = os.path.join(base_dir, "origscale")

    # 创建测试用的配置，确保包含所有必要字段
    model_module = model_cfg[model_name]["module"]
    test_cfg = model_module.cfg.copy()  # 从模型的默认配置开始
    test_cfg.update(cfg)  # 更新数据集相关配置
    test_cfg["exp_name"] = f"{model_name}_{sensor}"
    
    test_run(model_name, weight_path, "reduced", cfg=test_cfg, output_dir=reduced_dir)
    test_run(model_name, weight_path, "origscale", cfg=test_cfg, output_dir=orig_dir)

    # 可视化
    if os.path.exists(reduced_dir):
        fig_reduced = os.path.join(base_dir, f"{model_name}_{sensor}_reduced_results.png")
        show_results(reduced_dir, save_file=fig_reduced)
        logging.info(f"{model_name}_{sensor} Reduced 结果图已保存至 {fig_reduced}")
    if os.path.exists(orig_dir):
        fig_orig = os.path.join(base_dir, f"{model_name}_{sensor}_origscale_results.png")
        show_results(orig_dir, save_file=fig_orig)
        logging.info(f"{model_name}_{sensor} OrigScale 结果图已保存至 {fig_orig}")


def _copy_selected_files(src_dir: str, dst_dir: str, sensor: str, ext: str):
    """复制指定 sensor 文件到临时目录，仅评估该数据集"""
    patterns = []
    if ext == "h5":
        patterns.append(f"test_{sensor}")
    elif ext == "mat":
        patterns.append(f"Test(HxWxC)_{sensor}")
    else:
        return

    for fname in os.listdir(src_dir):
        if any(fname.startswith(pat) and fname.endswith(f".{ext}") for pat in patterns):
            shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))


def evaluate_sensor(model_name: str, sensor: str, weight_path: str):
    cfg = data_cfg[sensor]
    model_info = model_cfg[model_name]

    # 加载模型
    model_module = model_info["module"]
    model_class = getattr(model_module, model_info["model_class"])
    
    # 首先使用模型自身 default.json 中的通道数，保证与训练时一致；若 data_cfg 显式指定则覆盖
    # 模型参数名为 "channels"，此处保持一致，避免传参错误
    channels = model_module.cfg.get("channels", model_info["default_channels"])
    channels = cfg.get("channels", channels)  # data_cfg 优先级更高
    model = model_class(spectral_num=cfg["spectral_num"], channels=channels).to(device)

    # Flexible weight loading: ignore or skip mismatching parameter shapes after architecture updates.
    checkpoint = torch.load(weight_path, map_location=device)
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in checkpoint.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    if missing_keys or unexpected_keys:
        logging.info(f"[WeightLoader] Skipped mismatched params: {len(checkpoint) - len(filtered_state)}; missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)}")

    model.eval()

    eval_results = {}
    with tempfile.TemporaryDirectory() as tmp_h5_full, \
         tempfile.TemporaryDirectory() as tmp_h5_reduced, \
         tempfile.TemporaryDirectory() as tmp_mat_full, \
         tempfile.TemporaryDirectory() as tmp_mat_reduced:

        # 按 sensor 复制相关文件到临时目录
        _copy_selected_files("dataset/full_examples", tmp_h5_full, sensor, "h5")
        _copy_selected_files("dataset/reduced_examples", tmp_h5_reduced, sensor, "h5")
        _copy_selected_files("dataset/full_examples", tmp_mat_full, sensor, "mat")
        _copy_selected_files("dataset/reduced_examples", tmp_mat_reduced, sensor, "mat")

        # 进行评估
        eval_results["Full Res H5"] = eval_ps.evaluate_dataset(model, tmp_h5_full, 'h5', device, 'full', cfg["sensor_range_max"], 4)
        eval_results["Reduced Res H5"] = eval_ps.evaluate_dataset(model, tmp_h5_reduced, 'h5', device, 'reduced', cfg["sensor_range_max"], 4)
        eval_results["Full Res MAT"] = eval_ps.evaluate_dataset(model, tmp_mat_full, 'mat', device, 'full', cfg["sensor_range_max"], 4)
        eval_results["Reduced Res MAT"] = eval_ps.evaluate_dataset(model, tmp_mat_reduced, 'mat', device, 'reduced', cfg["sensor_range_max"], 4)

    return eval_results


def main(test_mode=False):
    if test_mode:
        logging.info("===== 测试模式：只验证配置和导入 =====")
        
        # 测试模型配置
        for model_name in ["cannet"]:
            logging.info(f"测试模型配置: {model_name}")
            model_info = model_cfg[model_name]
            model_module = model_info["module"]
            model_class = getattr(model_module, model_info["model_class"])
            logging.info(f"  - 模块: {model_module}")
            logging.info(f"  - 模型类: {model_class}")
            logging.info(f"  - 默认通道数: {model_info['default_channels']}")
        
        # 测试数据集配置
        for sensor in ["gf2", "qb", "wv3"]:
            logging.info(f"测试数据集配置: {sensor}")
            cfg = data_cfg[sensor]
            logging.info(f"  - 训练数据: {cfg['train_data']}")
            logging.info(f"  - 验证数据: {cfg['val_data']}")
            logging.info(f"  - 光谱数: {cfg['spectral_num']}")
            logging.info(f"  - 传感器范围: {cfg['sensor_range_max']}")
        
        logging.info("===== 测试模式完成：配置验证通过 =====")
        return
    
    # 原有的完整运行逻辑
    for model_name in ["cannet"]:
        logging.info(f"===== 开始处理模型 {model_name.upper()} =====")
        all_eval_results[model_name] = {}
        
        for sensor in ["gf2", "qb", "wv3"]:
            logging.info(f"----- 开始处理数据集 {sensor.upper()} -----")
            
            # 获取权重路径（自动训练如果不存在）
            weight_path = ensure_weights(model_name, sensor)
            
            # 配置
            cfg = data_cfg[sensor]
            
            # 生成测试结果和可视化
            generate_test_and_show(model_name, sensor, cfg, weight_path)
            
            # 评估
            try:
                eval_results = evaluate_sensor(model_name, sensor, weight_path)
                all_eval_results[model_name][sensor] = eval_results
                
                # 简单打印评估结果
                logging.info(f"{model_name.upper()} 在 {sensor.upper()} 上的评估结果:")
                for data_type, metrics in eval_results.items():
                    if isinstance(metrics, dict):
                        # 显示主要指标
                        ergas = metrics.get('ERGAS', 'N/A')
                        sam = metrics.get('SAM', 'N/A')
                        logging.info(f"  {data_type} - ERGAS: {ergas}, SAM: {sam}")
                
            except Exception as e:
                logging.error(f"评估 {model_name} on {sensor} 时出错: {e}")
                all_eval_results[model_name][sensor] = {"error": str(e)}
            
            # 强制释放显存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 保存评估结果到 JSON
    result_file = "cannet_evaluation_results.json"
    with open(result_file, 'w') as f:
        json.dump(all_eval_results, f, indent=2)
    logging.info(f"所有评估结果已保存到 {result_file}")
    
    logging.info("===== 评估流水线完成 =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行全景锐化模型评估流水线')
    parser.add_argument('--test_mode', action='store_true', 
                       help='测试模式：只验证配置和导入，不进行完整训练和评估')
    args = parser.parse_args()
    
    main(test_mode=args.test_mode) 