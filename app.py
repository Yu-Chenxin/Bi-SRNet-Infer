import gradio as gr
import os
import yaml
import torch
import numpy as np
from skimage import io
from torch.utils.data import DataLoader
import tempfile
from PIL import Image
import cv2

from models.SSCDl import SSCDl as Net
from cd_datasets import RS_ST as RS
from functools import partial

# Mod_RWKV
from Mod_RWKV.infer.worldmodel import Worldinfer
llm_path='./checkpoints/lm_weights/nonencoder'
encoder_path='./checkpoints/siglip2-base-patch16-384'
encoder_type='siglip' #[clip, whisper, siglip, speech]
mod_rwkv_model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

# img_path = './docs/03-Confusing-Pictures.jpg'
# image = Image.open(img_path).convert('RGB')
# text = '\x16User: Pleas discribe this image~\x17Assistant:'
# result,_ = mod_rwkv_model.generate(text, image)
# print(result)

# 初始化模型
def initialize_model(checkpoint_path):
    net = Net(3, RS.num_classes).cuda()
    net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    net.eval()
    return net

# 预测函数
def predict(model, img1, img2, selected_changes):
    # 创建临时目录保存中间文件
    with tempfile.TemporaryDirectory() as tmpdir:
        # 保存上传的图片
        img1_path = os.path.join(tmpdir, "img1.png")
        img2_path = os.path.join(tmpdir, "img2.png")
        img1.save(img1_path)
        img2.save(img2_path)
        
        # 创建测试数据集和加载器
        class TempDataset:
            def __init__(self, img1_path, img2_path):
                self.img1_path = img1_path
                self.img2_path = img2_path
                
            def __len__(self):
                return 1
                
            def __getitem__(self, idx):
                img1 = io.imread(self.img1_path)
                img2 = io.imread(self.img2_path)
                
                # 使用与pred_SCD.py相同的预处理方法
                img1 = RS.normalize_image(img1, 'A')
                img2 = RS.normalize_image(img2, 'B')
                
                from torchvision.transforms import functional as F
                return F.to_tensor(img1), F.to_tensor(img2)
                
            def get_mask_name(self, vi):
                return "result.png"
        
        # 创建数据集和加载器
        test_set = TempDataset(img1_path, img2_path)
        test_loader = DataLoader(test_set, batch_size=1)
        
        # 执行预测
        for vi, data in enumerate(test_loader):
            imgs_A, imgs_B = data
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            
            with torch.no_grad():
                # 基础预测
                out_change, outputs_A, outputs_B = model(imgs_A, imgs_B)
                out_change = torch.sigmoid(out_change)
                
                # 应用softmax到语义分割输出
                outputs_A = torch.softmax(outputs_A, dim=1)
                outputs_B = torch.softmax(outputs_B, dim=1)
                
                # 测试时增强 - 垂直翻转
                imgs_A_v = torch.flip(imgs_A, [2])
                imgs_B_v = torch.flip(imgs_B, [2])
                out_change_v, outputs_A_v, outputs_B_v = model(imgs_A_v, imgs_B_v)
                outputs_A_v = torch.flip(outputs_A_v, [2])
                outputs_B_v = torch.flip(outputs_B_v, [2])
                out_change_v = torch.flip(out_change_v, [2])
                outputs_A += torch.softmax(outputs_A_v, dim=1)
                outputs_B += torch.softmax(outputs_B_v, dim=1)
                out_change += torch.sigmoid(out_change_v)
                
                # 测试时增强 - 水平翻转
                imgs_A_h = torch.flip(imgs_A, [3])
                imgs_B_h = torch.flip(imgs_B, [3])
                out_change_h, outputs_A_h, outputs_B_h = model(imgs_A_h, imgs_B_h)
                outputs_A_h = torch.flip(outputs_A_h, [3])
                outputs_B_h = torch.flip(outputs_B_h, [3])
                out_change_h = torch.flip(out_change_h, [3])
                outputs_A += torch.softmax(outputs_A_h, dim=1)
                outputs_B += torch.softmax(outputs_B_h, dim=1)
                out_change += torch.sigmoid(out_change_h)
                
                # 测试时增强 - 垂直+水平翻转
                imgs_A_hv = torch.flip(imgs_A, [2, 3])
                imgs_B_hv = torch.flip(imgs_B, [2, 3])
                out_change_hv, outputs_A_hv, outputs_B_hv = model(imgs_A_hv, imgs_B_hv)
                outputs_A_hv = torch.flip(outputs_A_hv, [2, 3])
                outputs_B_hv = torch.flip(outputs_B_hv, [2, 3])
                out_change_hv = torch.flip(out_change_hv, [2, 3])
                outputs_A += torch.softmax(outputs_A_hv, dim=1)
                outputs_B += torch.softmax(outputs_B_hv, dim=1)
                out_change += torch.sigmoid(out_change_hv)
                
                # 平均所有增强结果
                outputs_A /= 4.0
                outputs_B /= 4.0
                out_change /= 4.0
                
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = out_change.cpu().detach() > 0.8
            change_mask = change_mask.squeeze()
            pred_A = torch.argmax(outputs_A, dim=1).squeeze()
            pred_B = torch.argmax(outputs_B, dim=1).squeeze()
            
            # 处理输出图像
            pred_A_masked = (pred_A * change_mask.long()).numpy()
            pred_B_masked = (pred_B * change_mask.long()).numpy()
            
            pred_A_masked_colored = RS.Index2Color(pred_A_masked)
            pred_B_masked_colored = RS.Index2Color(pred_B_masked)
            
            # 根据用户选择的变化类型进行增强处理
            if selected_changes and "all" not in selected_changes:
                # 创建特定变化类型的掩膜
                specific_change_mask = np.zeros_like(pred_A_masked, dtype=bool)
                for change_type in selected_changes:
                    if "->" in change_type:
                        from_class, to_class = change_type.split("->")
                        from_idx = RS.ST_CLASSES.index(from_class.strip())
                        to_idx = RS.ST_CLASSES.index(to_class.strip())
                        specific_change_mask |= ((pred_A_masked == from_idx) & (pred_B_masked == to_idx))
                
                # 对选定的变化类型进行增强处理
                enhanced_mask = specific_change_mask.astype(np.uint8) * 255
                
                # 去噪处理
                kernel = np.ones((3,3), np.uint8)
                enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel)
                enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel)
                
                # 膨胀操作使边缘更明显
                kernel = np.ones((5,5), np.uint8)
                enhanced_mask = cv2.dilate(enhanced_mask, kernel, iterations=1)
                
                # 查找轮廓并绘制边界
                contours, _ = cv2.findContours(enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 将轮廓绘制在原图上
                img1_array = np.array(img1)
                img2_array = np.array(img2)
                
                # 确保图像是三通道的
                if len(img1_array.shape) == 2:
                    img1_array = cv2.cvtColor(img1_array, cv2.COLOR_GRAY2RGB)
                if len(img2_array.shape) == 2:
                    img2_array = cv2.cvtColor(img2_array, cv2.COLOR_GRAY2RGB)
                
                # 在图像上绘制轮廓
                cv2.drawContours(img1_array, contours, -1, (0, 255, 0), 2)
                cv2.drawContours(img2_array, contours, -1, (0, 255, 0), 2)
                
                # 将处理后的图像转换回PIL格式
                processed_img1 = Image.fromarray(img1_array)
                processed_img2 = Image.fromarray(img2_array)
                
                # 使用处理后的图像替代原来的索引结果
                im1_index = processed_img1
                im2_index = processed_img2

                # 计算变化区域的像素数量
                change_pixels = np.sum(specific_change_mask)

            elif selected_changes and "all" in selected_changes:
                # 当选择"all"时，绘制所有变化类型的轮廓
                # 创建一个掩膜表示所有变化区域
                all_change_mask = change_mask.numpy().astype(np.uint8) * 255
                
                # 去噪处理
                kernel = np.ones((3,3), np.uint8)
                all_change_mask = cv2.morphologyEx(all_change_mask, cv2.MORPH_CLOSE, kernel)
                all_change_mask = cv2.morphologyEx(all_change_mask, cv2.MORPH_OPEN, kernel)
                
                # 膨胀操作使边缘更明显
                kernel = np.ones((5,5), np.uint8)
                all_change_mask = cv2.dilate(all_change_mask, kernel, iterations=1)
                
                # 查找轮廓并绘制边界
                contours, _ = cv2.findContours(all_change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 将轮廓绘制在原图上
                img1_array = np.array(img1)
                img2_array = np.array(img2)
                
                # 确保图像是三通道的
                if len(img1_array.shape) == 2:
                    img1_array = cv2.cvtColor(img1_array, cv2.COLOR_GRAY2RGB)
                if len(img2_array.shape) == 2:
                    img2_array = cv2.cvtColor(img2_array, cv2.COLOR_GRAY2RGB)
                
                # 在图像上绘制轮廓
                cv2.drawContours(img1_array, contours, -1, (255, 0, 0), 2)
                cv2.drawContours(img2_array, contours, -1, (255, 0, 0), 2)
                
                # 将处理后的图像转换回PIL格式
                processed_img1 = Image.fromarray(img1_array)
                processed_img2 = Image.fromarray(img2_array)
                
                # 使用处理后的图像替代原来的索引结果
                im1_index = processed_img1
                im2_index = processed_img2

                # 计算变化区域的像素数量
                change_pixels = np.sum(change_mask.numpy())

            else:
                # 默认情况，显示原始索引结果
                im1_index = Image.fromarray(pred_A_masked.astype('uint8'))
                im2_index = Image.fromarray(pred_B_masked.astype('uint8'))
                # 计算变化区域的像素数量
                change_pixels = np.sum(change_mask.numpy())
            
            # 直接返回PIL图像对象而不是文件路径
            im1_rgb = Image.fromarray(pred_A_masked_colored.astype('uint8'))
            im2_rgb = Image.fromarray(pred_B_masked_colored.astype('uint8'))
            

            
            # 返回图像对象和像素计数
            return [
                im1_rgb,
                im2_rgb,
                im1_index,
                im2_index,
                change_pixels
            ]

# 修改预测函数以处理来自不同tab的选择
def predict_with_selection(model, img1, img2, selected_all, selected_1, selected_2, selected_3, selected_4, selected_5, selected_6):
    # 合并所有选择
    all_selected = []
    for selections in [selected_all, selected_1, selected_2, selected_3, selected_4, selected_5, selected_6]:
        all_selected.extend(selections)
    # 去重
    all_selected = list(set(all_selected))
    return predict(model, img1, img2, all_selected)

# 计算实际面积的函数
def calculate_area(pixels, gsd):
    try:
        # 先判断是否传入了有效像素值
        if pixels is None or (isinstance(pixels, (int, float)) and pixels <= 0):
            return "错误：未执行变化检测"
        
        gsd_value = float(gsd)
        if gsd_value <= 0:
            return "错误：GSD必须大于0"
        
        area = pixels * (gsd_value ** 2)
        return f"变化区域面积：{area:.2f} 平方米 (GSD={gsd_value})"
    
    except ValueError:
        return "错误：请输入有效的GSD数值"

# 创建Gradio界面
def create_interface():
    # 初始化模型
    checkpoint_path = './checkpoints/best_checkpoint.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    model = initialize_model(checkpoint_path)
    
    # 定义接口
    def inference(img1, img2, selected_changes):
        try:
            result = predict(model, img1, img2, selected_changes)
            # 返回图像对象和像素计数
            return result
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # 返回4个空图像作为占位符和0像素计数
            return [None, None, None, None, 0]
    
    # 创建变化类型选项
    change_options = ["all"]
    for i in range(1, len(RS.ST_CLASSES)):  # 从1开始，跳过"unchanged"
        for j in range(1, len(RS.ST_CLASSES)):
            if i != j:
                change_options.append(f"{RS.ST_CLASSES[i]}->{RS.ST_CLASSES[j]}")
    
    # 创建界面
    with gr.Blocks(title="Bi-SRNet 变化检测", theme=gr.themes.Soft()) as demo:
        demo.queue()
        with gr.Tab("图像变化检测"):
            gr.Markdown("""
            <h1 style='text-align: center; color: #1a73e8;'>Bi-SRNet 遥感图像变化检测</h1>
            <p style='text-align: center; font-size: 16px;'>上传两个时期的遥感图像(T1和T2)，获取高精度变化检测结果</p>
            """)

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### 输入图像")
                    input_img1 = gr.Image(label="T1时期图像", type="pil", height=300, elem_id="input_img1")
                    input_img2 = gr.Image(label="T2时期图像", type="pil", height=300, elem_id="input_img2")
                    run_button = gr.Button("🚀 执行变化检测", variant="primary")

                with gr.Column(scale=2, min_width=500):
                    gr.Markdown("### 检测结果")
                    with gr.Row():
                        with gr.Column():
                            output_im1_rgb = gr.Image(label="时期1 RGB结果", interactive=False, height=300, elem_id="output_im1_rgb")
                            output_im2_rgb = gr.Image(label="时期2 RGB结果", interactive=False, height=300, elem_id="output_im2_rgb")
                        with gr.Column():
                            output_im1_index = gr.Image(label="时期1 索引结果", interactive=False, height=300, elem_id="output_im1_index")
                            output_im2_index = gr.Image(label="时期2 索引结果", interactive=False, height=300, elem_id="output_im2_index")


            # 添加变化类型选择模块
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### 变化类型选择")
                gr.Markdown("选择要重点分析的变化类型，或保持默认分析所有变化")

                # 按起始类别分组
                with gr.Tabs():
                    with gr.TabItem("全部"):
                        change_selector = gr.CheckboxGroup(
                            choices=["all"],
                            value=["all"],
                            label="全选"
                        )

                    # 为每个类别创建单独的tab
                    with gr.TabItem(RS.ST_CLASSES[1]):
                        change_selector_1 = gr.CheckboxGroup(
                            choices=[opt for opt in change_options if opt.startswith(RS.ST_CLASSES[1]+"->")],
                            value=[],
                            label=f"从'{RS.ST_CLASSES[1]}'的变化"
                        )

                    with gr.TabItem(RS.ST_CLASSES[2]):
                        change_selector_2 = gr.CheckboxGroup(
                            choices=[opt for opt in change_options if opt.startswith(RS.ST_CLASSES[2]+"->")],
                            value=[],
                            label=f"从'{RS.ST_CLASSES[2]}'的变化"
                        )

                    with gr.TabItem(RS.ST_CLASSES[3]):
                        change_selector_3 = gr.CheckboxGroup(
                            choices=[opt for opt in change_options if opt.startswith(RS.ST_CLASSES[3]+"->")],
                            value=[],
                            label=f"从'{RS.ST_CLASSES[3]}'的变化"
                        )

                    with gr.TabItem(RS.ST_CLASSES[4]):
                        change_selector_4 = gr.CheckboxGroup(
                            choices=[opt for opt in change_options if opt.startswith(RS.ST_CLASSES[4]+"->")],
                            value=[],
                            label=f"从'{RS.ST_CLASSES[4]}'的变化"
                        )

                    with gr.TabItem(RS.ST_CLASSES[5]):
                        change_selector_5 = gr.CheckboxGroup(
                            choices=[opt for opt in change_options if opt.startswith(RS.ST_CLASSES[5]+"->")],
                            value=[],
                            label=f"从'{RS.ST_CLASSES[5]}'的变化"
                        )

                    with gr.TabItem(RS.ST_CLASSES[6]):
                        change_selector_6 = gr.CheckboxGroup(
                            choices=[opt for opt in change_options if opt.startswith(RS.ST_CLASSES[6]+"->")],
                            value=[],
                            label=f"从'{RS.ST_CLASSES[6]}'的变化"
                        )

                clear_button = gr.Button("❌ 清除", variant="primary", elem_id="clear_button")

            # 添加面积计算组件
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### 面积计算")
                with gr.Row():
                    gsd_input = gr.Textbox(label="GSD (Ground Sample Distance)", placeholder="输入GSD值(m/pixel)")
                    area_output = gr.Textbox(label="计算结果", interactive=False)
                    calculate_button = gr.Button("📊 计算面积", variant="primary")
                pixel_count = gr.State()  # 存储像素计数的隐藏状态

            # 绑定清除按钮事件
            def clear_selections():
                return [
                    ["all"],  # 主选择器
                    [],  # change_selector_1
                    [],  # change_selector_2
                    [],  # change_selector_3
                    [],  # change_selector_4
                    [],  # change_selector_5
                    [],  # change_selector_6
                ]

            clear_button.click(
                fn=clear_selections,
                inputs=[],
                outputs=[
                    change_selector,
                    change_selector_1,
                    change_selector_2,
                    change_selector_3,
                    change_selector_4,
                    change_selector_5,
                    change_selector_6
                ]
            )

            # 绑定面积计算按钮事件
            calculate_button.click(
                fn=calculate_area,
                inputs=[pixel_count, gsd_input],
                outputs=[area_output]
            )


            with gr.Accordion("📂 示例图像", open=False):
                with gr.Row(equal_height=True):
                    for i in range(1, 7):
                        with gr.Column(scale=1, min_width=200):
                            # 构造路径
                            img1_path = f"example/example{i}/t1.png"
                            img2_path = f"example/example{i}/t2.png"

                            try:
                                img1 = Image.open(img1_path)
                                img2 = Image.open(img2_path)
                            except Exception as e:
                                print(f"Error loading example {i}: {e}")
                                img1 = None
                                img2 = None

                            # 显示图像
                            with gr.Row():
                                if img1 is not None:
                                    gr.Image(value=img1, label=f"T1 - 示例{i}", height=100, interactive=False)
                                else:
                                    gr.Image(label=f"T1-示例{i}", height=100, interactive=False)

                                if img2 is not None:
                                    gr.Image(value=img2, label=f"T2 - 示例{i}", height=100, interactive=False)
                                else:
                                    gr.Image(label=f"T2-示例{i}", height=100, interactive=False)

                            # 加载按钮
                            load_btn = gr.Button(f"📥 加载示例{i}", variant="primary")

                            # 定义加载函数（闭包处理 i 的值）
                            def load_example(idx):
                                try:
                                    img1 = Image.open(f"example/example{idx}/t1.png")
                                    img2 = Image.open(f"example/example{idx}/t2.png")
                                    return [img1, img2]
                                except Exception as e:
                                    print(f"Error loading example {idx}: {e}")
                                    return [None, None]

                            # 绑定事件
                            load_btn.click(
                                fn=partial(load_example, idx=i),
                                inputs=[],
                                outputs=[input_img1, input_img2]
                            )

            # 添加使用说明和输出解释
            with gr.Accordion("🔍 查看详细说明", open=False):
                gr.Markdown("""
                ## 使用方法
                1. 在左侧上传两个不同时期的遥感图像
                2. （可选）选择关注的变化类型，默认分析所有变化
                3. 点击"执行变化检测"按钮
                4. 查看右侧生成的变化检测结果

                ## 输出说明
                - **RGB结果**: 显示变化区域的彩色可视化结果
                - **索引结果**: 如果选择了特定变化类型，则显示增强后带有标注的结果；否则显示标准索引结果
                - 图像经过测试时增强(TTA)处理，提高检测精度
                - 使用与训练一致的预处理和阈值(0.8)

                ## 类型选择
                - 选择特定的变化类型以重点关注
                - 默认情况下(all)，所有变化类型都显示
                - 取消所有选择以展示索引图像

                ## 面积计算
                - 输入图像的地面采样距离(GSD)值
                - 点击"计算面积"按钮获得实际面积
                - 面积计算公式：像素数 × GSD²

                ## 技术特点
                - 基于Bi-SRNet深度学习模型
                - 采用双时相语义推理机制
                - 支持高分辨率遥感图像分析
                """)

            run_button.click(
                fn=partial(predict_with_selection, model),
                inputs=[input_img1, input_img2, change_selector, change_selector_1, change_selector_2, change_selector_3, change_selector_4, change_selector_5, change_selector_6],
                outputs=[output_im1_rgb, output_im2_rgb, output_im1_index, output_im2_index, pixel_count]
            ).then(
                fn=calculate_area,
                inputs=[pixel_count, gsd_input],
                outputs=area_output
            )
            







        with gr.Tab("RWKV LM Assistant"):
            gr.Markdown("""
            <h1 style='text-align: center; color: #1a73e8;'>RWKV 语言模型助手</h1>
            <p style='text-align: center; font-size: 16px;'>与RWKV语言模型进行对话。RWKV是一个开源的大语言模型，具有优秀的性能和效率。</p>
            """)

            with gr.Row():
                # 左侧添加图像输入区域
                with gr.Column(scale=2):
                    rwkv_image_input = gr.Image(type="pil", label="上传图片", height=300)
                    image_description = gr.Textbox(label="图像描述", interactive=False, lines=8, max_lines=10)
                    auto_describe_checkbox = gr.Checkbox(label="自动使用图像描述作为上下文", value=True)

                    def generate_image_description(image):
                        if image is not None:
                            text = '\x16User: Please describe this image\x17Assistant:'
                            result, _ = mod_rwkv_model.generate(text, image)
                            return result
                        return ""

                    # 当图像上传时自动生成描述
                    rwkv_image_input.change(
                        fn=generate_image_description,
                        inputs=[rwkv_image_input],
                        outputs=[image_description]
                    )

                # 右侧保持原有聊天界面
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height="60vh",
                        label="对话记录",
                        elem_id="chatbot-container",
                        type="messages",
                        render_markdown=True
                    )       
                    with gr.Row():
                        msg = gr.Textbox(label="输入消息", placeholder="请输入您的问题...", scale=9)
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                    with gr.Row():
                        clear_btn = gr.Button("清空对话")
                        examples = gr.Examples(
                            examples=["RWKV是什么，和Transfomer相比有什么区别?", "什么是Bi-SRNet", "什么是遥感图像变化检测", "在遥感领域GSD是什么，怎么计算"],
                            inputs=msg
                        )

            # 修改RWKV API调用函数
            def rwkv_chat(message, history, image, image_desc, use_auto_desc):
                import requests
                import json

                # API配置
                api_url = "http://127.0.0.1:8000/v4/chat/completions"

                # 构建消息内容
                with open('doc/chat_material.yaml', 'r', encoding='utf-8') as f:
                    corpora = yaml.safe_load(f)

                # 关键词映射到语料
                keyword_to_corpus = {
                    "bisrnet": ["bisrnet", "bi-srnet", "sscd", "语义变化检测", "双时相语义"],
                    "rs_basic": ["遥感", "rs", "ground sample distance", "gsd", "光谱"],
                    "rwkv":["rwkv","rnn","lm","线性注意力","receptance weighted key value", "RWKV"],
                    "general": []  # 默认语料
                }

                # 根据用户问题选择合适的语料
                def select_corpus(user_question):
                    user_question_lower = user_question.lower()
                    for corpus_key, keywords in keyword_to_corpus.items():
                        for keyword in keywords:
                            if keyword.lower() in user_question_lower:
                                return corpora[corpus_key]
                    # 默认返回general语料
                    return corpora["general"]

                # 动态选择语料
                selected_corpus = select_corpus(message)

                # 构建消息列表
                messages = [{"role": "user", "content": selected_corpus}]

                # 如果有图像且启用自动描述，则将其添加到上下文中
                if image is not None and use_auto_desc and image_desc:
                    messages.append({"role": "user", "content": f"图像描述信息: {image_desc}"})

                messages.append({"role": "user", "content": message})

                # 请求参数
                payload = {
                    "messages": messages,  # 只发送当前消息
                    "max_tokens": 8192,
                    "stop_tokens": [0, 261, 24281],
                    "temperature": 1.0,
                    "noise": 1.5,
                    "stream": True,
                    "enable_think": True,
                    "chunk_size": 8
                }

                headers = {
                    "Content-Type": "application/json"
                }

                try:
                    response = requests.post(api_url, data=json.dumps(payload), headers=headers, stream=True)
                    response.raise_for_status()

                    full_response = ""
                    thinking_content = ""
                    answer_content = ""
                    thinking_finished = False

                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]  # 移除"data: "前缀
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        full_response += content

                                        # 处理思考过程和正文的分离
                                        if not thinking_finished:
                                            if "</think>" in content:
                                                # 找到思考结束标记
                                                parts = content.split("</think>", 1)
                                                thinking_content += parts[0]
                                                if len(parts) > 1:
                                                    answer_content += parts[1]
                                                thinking_finished = True
                                            else:
                                                # 仍在思考阶段
                                                thinking_content += content
                                        else:
                                            # 思考已完成，添加到答案内容
                                            answer_content += content

                                        # 构造显示内容：思考过程(如果有) + 正文
                                        display_content = ""
                                        if thinking_content or "</think>" in full_response:
                                            # 如果有思考内容或者已经结束思考，显示思考过程
                                            if thinking_content.strip():
                                                # 添加 open 属性使 details 默认展开
                                                display_content += f"<details open><summary>思考过程</summary>\n\n{thinking_content}\n\n</details>\n\n"
                                            display_content += "---\n"

                                        display_content += answer_content

                                        # 实时更新聊天界面，保持输入框内容不变
                                        # 保留完整历史记录在UI上显示
                                        yield message, history + [{"role": "user", "content": message}, {"role": "assistant", "content": display_content}]
                                except json.JSONDecodeError:
                                    continue
                                
                    # 最终完整响应，清空输入框
                    # 同样处理最终响应
                    display_content = ""
                    if thinking_content or "</think>" in full_response:
                        if thinking_content.strip():
                            # 添加 open 属性使 details 默认展开
                            display_content += f"<details open><summary>思考过程</summary>\n\n{thinking_content}\n\n</details>\n\n"
                        display_content += "---\n"

                    display_content += answer_content

                    yield "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": display_content}]
                except Exception as e:
                    error_msg = f"请求失败: {str(e)}"
                    yield "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}]

            # 更新事件绑定
            msg.submit(rwkv_chat, [msg, chatbot, rwkv_image_input, image_description, auto_describe_checkbox], [msg, chatbot], queue=True) 
            send_btn.click(rwkv_chat, [msg, chatbot, rwkv_image_input, image_description, auto_describe_checkbox], [msg, chatbot], queue=True) 
            clear_btn.click(lambda: None, None, chatbot, queue=False)

    return demo

# 主函数
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, show_api=False)