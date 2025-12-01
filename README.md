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

- CUDA 12.6+
- Python 3.12
- PyTorch 2.8.0+
- torchvision
- numpy
- opencv-python
- pillow

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

## 使用方法

```bash
python app.py
```
- 在浏览器中打开 http://localhost:7860
- 如果在浏览器中打开失败，请检查端口是否被占用


