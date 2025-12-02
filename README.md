# Bi-SRNet-Infer

使用Gradio为Bi-SRNet创建了一个可视化界面，顺便带了一些关于RWKV的小私货

## 项目结构

```
Bi-SRNet-Infer/
    ├── Mod_RWKV/      # RWKV相关模块
    │   ├── infer/     # 推理相关实现
    │   │   └── rwkv/  # RWKV核心实现
    │   └── lg_train/  # 训练相关模块
    ├── cd_datasets/   # 变化检测数据集处理
    ├── checkpoints/   # 模型检查点文件
    ├── doc/           # 文档材料
    ├── models/        # 模型定义
    │   ├── BiSRNet.py # BiSRNet主模型
    │   └── SSCDl.py   # SSCDl模型
    ├── utils/         # 工具函数库
    └── app.py         # 主应用入口
```

## 功能特性

- 基于Bi-SRNet模型的遥感图像变化检测（后训练模型注重耕地变化检测）
- 支持自定义重点监测的变化类别（例如ground->building；默认为all； 不勾选将展示索引图像）
- 通过GSD的计算占地面积（GSD通过传感器高度、像元尺寸、焦距等参数通过计算获得） 
- 集成RWKV大语言模型支持（通过prompt注入，实现专业的语言理解）
- 通过 Mod RWKV 为 RWkV LLM 提供视觉理解支持

## 环境依赖

- CUDA 12.6+ / ROCm 6.4+
- Python 3.12
- PyTorch 2.8.0+ (Supports CUDA & ROCm)
- torchvision
- numpy
- opencv-python
- pillow
- gradio
- transformers
- ninja

## 安装说明

```bash
git clone https://github.com/Yu-Chenxin/Bi-SRNet-Infer.git
cd Bi-SRNet-Infer
pip install -r requirements.txt
```
（requirements.txt正在赶工中）

## 权重下载

**ModRWKV 权重**
```bash
wget https://huggingface.co/ZoomFly/rwkvsee0.4B/resolve/main/nonencoder.pth
```

**Siglip2 权重**
```bash
wget https://huggingface.co/google/siglip2-base-patch16-384/resolve/main/model.safetensors
```
- 你需要将这两个权重放在以下位置：

```
Bi-SRNet-Infer/
    └── checkpoints/
        ├── mod_rwkv/
        │    └── nonencoder.pth    # ModRWKV 权重放置位置
        └── siglip2/
            └── model.safetensors  # Siglip2 权重放置位置
```

## RWKV LLM 大语言模型使用方法

- 对于 RWKV LLM 大语言模型，我们使用了社区开发者[**@Alic-Li**](https://github.com/Alic-Li/)的后端[**rwkv_lightning**](https://github.com/RWKV-Vibe/rwkv_lightning)

- 具体部署详情请访问[**rwkv_lightning**](https://github.com/RWKV-Vibe/rwkv_lightning)的Readme

- 请使用带有CUDA graph支持的API启动推理后端来获得更好的性能
```bash
python single_infer.py --model-path <your model path> --port <your port number>
```

- 若未给出 **port number** 后端默认运行在端口8000上

- 代码默认后端在同一台机器上运行，如若要分开在多台机器上运行，请修改**app.py**中的第586行：

```python
586    api_url = "http://127.0.0.1:8000/v4/chat/completions" 
```

- 请将127.0.0.1修改为实际IP
- 若后端修改了端口，请将8000修改为实际端口

## 使用方法

```bash
python app.py
```
- 在浏览器中打开 http://localhost:7860
- 如果在浏览器中打开失败，请检查端口是否被占用
- 如要在其他机器端访问，请将localhost修改为实际IP

## 特别鸣谢

- RWKV架构作者[**@PENG Bo**](https://github.com/BlinkDL), 项目[**RWKV-LM**](https://github.com/BlinkDL/RWKV-LM)

- Bi-SRNet模型作者[**@Lei**](https://github.com/DingLei14)，项目[**Bi-SRNet**](https://github.com/DingLei14/Bi-SRNet)

- 社区开发者[**@Alic-Li**](https://github.com/Alic-Li/)，项目[**rwkv_lightning**](https://github.com/RWKV-Vibe/rwkv_lightning)
