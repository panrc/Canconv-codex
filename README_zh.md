# CANConv 项目中文使用指南

本项目是论文 "Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening" 的官方实现。

## 目录
1. [环境设置与编译](#1-环境设置与编译)
2. [数据集准备](#2-数据集准备)
3. [训练模型](#3-训练模型)
4. [测试模型与生成结果](#4-测试模型与生成结果)
5. [可视化测试结果](#5-可视化测试结果)
6. [查看 .mat 结果文件内容](#6-查看-mat-结果文件内容)
7. [其他脚本说明](#7-其他脚本说明)

---

## 1. 环境设置与编译

### a. 依赖安装
确保您的开发环境满足基本要求，例如 Python、PyTorch 和 CUDA（如果使用 GPU）。

主要的 Python 依赖项列在 `requirements.txt` 文件中。您可以通过以下命令在激活的 Python 虚拟环境（推荐）中安装它们：
```bash
# 激活您的虚拟环境 (例如 .venv)
# source .venv/bin/activate  (Linux/macOS)
# .venv\Scripts\activate (Windows)

python -m pip install -r requirements.txt
```
如果遇到 `externally-managed-environment` 错误，请确保您使用的是虚拟环境中的 `pip`。

### b. 编译项目
项目包含一些需要编译的C++/CUDA代码。运行以下脚本来编译：
```bash
bash ./build.sh
```
此脚本会执行以下操作：
1.  使用 `pip` 安装 `requirements.txt` 中的依赖。
2.  初始化并更新 Git 子模块（例如 `kmcuda`）。
3.  使用 CMake 和 Ninja 配置并构建项目。

**常见编译问题处理：**
*   **`python: command not found`**: `build.sh` 脚本默认使用 `python`。如果您的系统中 `python` 命令未链接到 `python3`，或者您希望明确使用 `python3`，可以编辑 `build.sh` 将 `python` 改为 `python3`。
*   **CMake 路径错误**: 如果遇到 CMake 抱怨路径不一致的问题，通常是因为之前在不同路径下生成过 CMake 缓存。删除 `build/` 目录可以解决此问题：
    ```bash
    rm -rf build
    ```
    然后重新运行 `./build.sh`。
*   **CUDA 编译器问题**:
    *   确保 CUDA Toolkit 已正确安装，并且 `nvcc` 在您的 `PATH` 环境变量中。
    *   检查 `CUDA_HOME` 或 `CUDA_PATH` 环境变量是否设置正确。
*   **`ModuleNotFoundError: No module named 'canconv.layers.kmeans.libKMCUDA'`**:
    *   此错误表示 `kmcuda` 子模块的共享库（通常是 `libKMCUDA.so`）未被正确编译或放置到 `canconv/layers/kmeans/` 目录下。正常情况下，`build.sh` 应该会自动处理。如果遇到此问题，请确保 `build.sh` 完整执行无误。

---

## 2. 数据集准备

*   项目使用的数据集主要来自 [liangjiandeng/PanCollection](https://github.com/liangjiandeng/PanCollection)。
*   数据集的预设配置（包括路径、光谱通道数、缩放因子等）定义在 `presets.json` 文件中。
    例如，`wv3` 数据集的配置可能如下：
    ```json
    {
        "wv3": {
            "train_data": "/datasets/wv3/train_wv3.h5",
            "val_data": "/datasets/wv3/valid_wv3.h5",
            "test_reduced_data": "/datasets/wv3/test_wv3_multiExm1.h5",
            "test_origscale_data": "/datasets/wv3/test_wv3_OrigScale_multiExm1.h5",
            "spectral_num": 8,
            "dataset_scale": 2047
        },
        // ... 其他数据集如 gf2, qb ...
    }
    ```
*   **您需要根据 `presets.json` 中定义的路径，将下载的数据集（`.h5` 文件）放置在您系统中的相应位置。** 例如，您可以创建一个名为 `datasets` 的文件夹在项目根目录下，然后在其中按数据集名称 (wv3, gf2 等) 创建子文件夹存放数据。如果您的实际数据路径与 `presets.json` 中的不符，请相应修改 `presets.json` 文件。

---

## 3. 训练模型

训练模型使用 `canconv/scripts/train.py` 脚本。

**基本命令格式：**
```bash
python -m canconv.scripts.train <model_name> <dataset_preset>
```
或者，如果您的环境默认 `python` 不是虚拟环境中的 Python 3：
```bash
python3 -m canconv.scripts.train <model_name> <dataset_preset>
```

**参数说明：**
*   `<model_name>`: 要训练的模型名称，例如 `cannet`。模型定义在 `canconv/models/` 目录下。
*   `<dataset_preset>`: 在 `presets.json` 中定义的数据集预设名称，例如 `wv3`, `gf2`, `qb`。

**示例：**
使用 `cannet` 模型和 `wv3` 数据集进行训练：
```bash
python3 -m canconv.scripts.train cannet wv3
```

**GPU 使用：**
*   默认情况下，代码会尝试使用 `cuda:0`。
*   如果需要使用特定的 GPU，可以通过设置 `CUDA_VISIBLE_DEVICES` 环境变量来指定。例如，使用第二个 GPU (索引为 1)：
    ```bash
    CUDA_VISIBLE_DEVICES=1 python3 -m canconv.scripts.train cannet wv3
    ```

**训练输出：**
*   训练过程中的日志、模型权重（ checkpoints）和 TensorBoard 日志通常会保存在项目根目录下的 `runs/` 文件夹中，并以实验名称（通常是模型名+数据集预设名）组织的子目录中。

---

## 4. 测试模型与生成结果

测试模型并生成超分辨率结果使用 `canconv/scripts/test.py` 脚本。

**基本命令格式：**
```bash
python3 -m canconv.scripts.test <model_name> <weight_file_path> --preset <dataset_preset> [其他可选参数]
```
或者，如果您想直接指定测试数据集的 `.h5` 文件路径，而不是通过预设中的 `test_reduced_data` 或 `test_origscale_data`：
```bash
python3 -m canconv.scripts.test <model_name> <weight_file_path> <test_h5_file_path> --preset <dataset_preset_for_scale_etc>
```
**注意：** 即使直接提供测试文件路径，通常仍需提供 `--preset` 参数，因为脚本会从中读取如 `dataset_scale` 等重要配置。

**参数说明：**
*   `<model_name>`: 模型的名称，例如 `cannet`。
*   `<weight_file_path>`: 预训练或您自己训练好的模型权重文件（`.pth` 文件）的路径。预训练权重通常位于 `weights/` 目录下 (例如 `weights/cannet_wv3.pth`)。
*   `<test_h5_file_path>` (可选，替代预设中的测试路径): 直接指定测试用的 `.h5` 数据集文件路径。
*   `--preset <dataset_preset>`: **必需。** 指定用于测试的数据集预设名称（来自 `presets.json`）。这主要用于加载与该数据集相关的配置，如 `dataset_scale`、光谱通道数以及预设的测试文件路径（如果未直接提供 `<test_h5_file_path>`）。
*   `test_dataset` (位置参数，当不使用 `--preset` 中的测试路径时): 可以是 `reduced` 或 `origscale` 来使用预设中定义的缩减分辨率或原始分辨率测试集，也可以是 `.h5` 文件的直接路径。
    *   当您为 `test_dataset` 提供一个直接的 `.h5` 路径时，例如：
        `python3 -m canconv.scripts.test cannet weights/cannet_wv3.pth dataset/your_test_data/your_test_file.h5 --preset wv3`
        这里的 `--preset wv3` 仍然重要，因为它会为 `wv3` 数据类型提供 `dataset_scale` 等元数据。

**示例：**
使用 `cannet_wv3.pth` 权重，在 `wv3` 数据集的 "原始分辨率" 测试集上进行测试：
```bash
python3 -m canconv.scripts.test cannet weights/cannet_wv3.pth origscale --preset wv3
```
或者，如果您有一个自定义的测试文件 `my_test_data.h5`，并且它符合 `wv3` 数据集的特性 (例如波段数、数据范围等):
```bash
# 假设 my_test_data.h5 位于 dataset/custom/ 目录下
python3 -m canconv.scripts.test cannet weights/cannet_wv3.pth dataset/custom/my_test_data.h5 --preset wv3
```

**测试输出：**
*   生成的超分辨率图像结果会以 `.mat` 文件的形式保存。
*   默认的输出目录是权重文件所在目录下的一个与权重文件名相关的子目录。例如，如果权重是 `weights/cannet_wv3.pth`，输出的 `.mat` 文件可能在 `weights/cannet_wv3/results/` 目录下，通常命名为 `output_mulExm_0.mat`, `output_mulExm_1.mat` 等。

**常见测试问题：**
*   **`KeyError: 'dataset_scale'`**: 这通常意味着脚本未能从配置中加载 `dataset_scale`。**请确保您在使用 `test.py` 时总是提供了 `--preset <dataset_preset>` 参数**，即使您直接指定了测试文件的路径。

---

## 5. 可视化测试结果

使用 `canconv/scripts/show.py` 脚本可以将一批测试生成的 `.mat` 文件中的图像结果可视化为一个图集。

**基本命令格式：**
```bash
# 直接在屏幕上显示图集
python3 -m canconv.scripts.show <path_to_results_parent_directory_or_results_dir>

# 将图集保存到文件
python3 -m canconv.scripts.show <path_to_results_parent_directory_or_results_dir> <output_image_filename.png>
```

**参数说明：**
*   `<path_to_results_parent_directory_or_results_dir>`: 指向包含 `.mat` 结果的目录。
    *   它可以是 `results` 目录的父目录（例如 `weights/cannet_wv3/`，脚本会自动查找其下的 `results` 子目录）。
    *   也可以直接是 `results` 目录本身（例如 `weights/cannet_wv3/results/`）。
*   `<output_image_filename.png>` (可选): 如果提供此参数，则会将可视化图集保存为指定的图像文件。如果不提供，则会直接在屏幕上显示。

**脚本内部逻辑：**
脚本会尝试加载指定目录下名为 `output_mulExm_0.mat` 到 `output_mulExm_19.mat` 的20个文件，提取其中的 `"sr"` (超分辨率图像) 变量，将其转换为 RGB 图像，并以 4x5 的网格形式显示或保存。

**示例：**
显示 `weights/cannet_wv3/results/` 目录下的测试结果：
```bash
python3 -m canconv.scripts.show weights/cannet_wv3/results/
```
或者，从 `weights/cannet_wv3/` 自动查找 `results` 子目录：
```bash
python3 -m canconv.scripts.show weights/cannet_wv3/
```
将上述结果保存到 `wv3_overview.png`：
```bash
python3 -m canconv.scripts.show weights/cannet_wv3/results/ wv3_overview.png
```

**常见可视化问题：**
*   **路径错误 (`FileNotFoundError` 或 `NotADirectoryError`)**: 确保您提供的 `<path_to_results_parent_directory_or_results_dir>` 是正确的，并且该目录下（或其 `results` 子目录下）确实存在 `output_mulExm_*.mat` 文件。

---

## 6. 查看 .mat 结果文件内容

如果您想单独查看某个 `.mat` 文件的具体内容（包含哪些变量，它们的形状和数据类型等），或者想用 GUI 工具查看图像：

*   **使用 MATLAB 或 Octave**:可以直接打开和浏览 `.mat` 文件。
*   **使用 Python (Scipy)**:
    ```python
    import scipy.io
    mat_data = scipy.io.loadmat('path_to_your_file.mat')
    print(mat_data.keys()) # 查看所有变量名
    # sr_image = mat_data['sr'] # 假设超分辨率图像存储在 'sr' 变量中
    ```
*   **使用之前提供的 `mat_viewer_gui.py` 脚本** (如果您还保留着它或者想重新创建): 这个脚本提供了一个简单的图形界面来加载 `.mat` 文件，列出变量，并尝试显示图像。运行它需要 `tkinter` 和 `matplotlib`。
    ```bash
    # 确保已安装 tkinter (sudo apt install python3-tk) 和 matplotlib (pip install matplotlib)
    python3 mat_viewer_gui.py
    ```

---

## 7. 其他脚本说明

*   `canconv/scripts/summary.py`: 从脚本名推测，此脚本可能用于生成训练过程的摘要或从 TensorBoard 日志中提取关键信息。具体用法可能需要查看其代码或运行 `python3 -m canconv.scripts.summary --help`。

---

希望这份中文指南能帮助您更好地使用本项目！ 