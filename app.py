import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import gradio as gr
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, SUPPORTED_ARCHITECTURES, DEFAULT_EPSILON
from src.models import ModelManager
from src.datasets import DatasetManager
from src.attacks import FGSM, PGD, CarliniWagner, DeepFool, AutoAttack, SquareAttack
from src.metrics import RobustnessMetrics, QualityMetrics
from src.defenses import DefenseMethods
from src.visualization import Visualizer
from src.utils import (
    ExperimentManager,
    PresetManager,
    TransferabilityAnalyzer,
    normalize_image,
    image_to_tensor,
    tensor_to_image,
    compute_perturbation_metrics,
)


model_manager = ModelManager()
dataset_manager = DatasetManager()
preset_manager = PresetManager()
experiment_manager = ExperimentManager()
robustness_metrics = RobustnessMetrics(model_manager)
quality_metrics = QualityMetrics()
defense_methods = DefenseMethods()
visualizer = Visualizer()


ATTACK_METHODS = {
    "fgsm": {"name": "FGSM", "class": FGSM, "type": "whitebox"},
    "pgd": {"name": "PGD", "class": PGD, "type": "whitebox"},
    "cw": {"name": "C&W (Carlini-Wagner)", "class": CarliniWagner, "type": "whitebox"},
    "deepfool": {"name": "DeepFool", "class": DeepFool, "type": "whitebox"},
    "autoattack": {"name": "AutoAttack", "class": AutoAttack, "type": "whitebox"},
    "square": {"name": "Square Attack (Black-box)", "class": SquareAttack, "type": "blackbox"},
}


def get_attack_instance(attack_name: str, model, params: Dict[str, Any]):
    attack_class = ATTACK_METHODS[attack_name]["class"]
    return attack_class(model, **params)


def get_model_choices():
    models = model_manager.list_models()
    return [(f"{m.name} ({m.architecture})", m.id) for m in models]


def get_dataset_choices():
    datasets = dataset_manager.list_datasets()
    return [(f"{d.name} ({d.total_images}张)", d.id) for d in datasets]


def upload_model(file, name, architecture, is_state_dict, labels_file):
    if not file:
        return None, "请上传模型文件"
    if not name:
        return None, "请输入模型名称"
    if not architecture:
        return None, "请选择模型架构"

    try:
        labels_path = labels_file.name if labels_file else None
        model_info = model_manager.upload_model(
            file.name, name, architecture, is_state_dict, labels_path
        )
        return (
            f"模型上传成功！\nID: {model_info.id}\n名称: {model_info.name}\n"
            f"架构: {model_info.architecture}\n类别数: {model_info.num_classes}\n"
            f"输入尺寸: {model_info.input_size}x{model_info.input_size}"
        ), refresh_model_list()
    except Exception as e:
        return f"上传失败: {str(e)}", refresh_model_list()


def delete_model(model_id):
    if model_manager.delete_model(model_id):
        return "模型删除成功", refresh_model_list()
    return "删除失败", refresh_model_list()


def refresh_model_list():
    models = model_manager.list_models()

    data = []
    for m in models:
        clean_acc = f"{m.clean_accuracy:.2f}%" if m.clean_accuracy is not None else "未评估"
        data.append([
            m.name, m.architecture, m.num_classes,
            m.upload_time, clean_acc, m.id
        ])

    df = pd.DataFrame(data, columns=[
        "名称", "架构", "类别数", "上传时间", "Clean Accuracy", "ID"
    ])
    return df


def upload_dataset(file, name):
    if not file:
        return None, "请上传数据集ZIP文件"
    if not name:
        return None, "请输入数据集名称"

    try:
        ds_info = dataset_manager.upload_dataset(file.name, name)
        class_dist = "\n".join([f"{k}: {v}张" for k, v in ds_info.class_distribution.items()])
        size_range = f"{ds_info.image_size_range[0]}x{ds_info.image_size_range[1]} ~ {ds_info.image_size_range[2]}x{ds_info.image_size_range[3]}"
        return (
            f"数据集上传成功！\nID: {ds_info.id}\n名称: {ds_info.name}\n"
            f"总图片数: {ds_info.total_images}\n尺寸范围: {size_range}\n"
            f"类别分布:\n{class_dist}"
        ), refresh_dataset_list()
    except Exception as e:
        return f"上传失败: {str(e)}", refresh_dataset_list()


def refresh_dataset_list():
    datasets = dataset_manager.list_datasets()

    data = []
    for d in datasets:
        data.append([
            d.name, d.total_images, len(d.class_distribution),
            d.upload_time, "是" if d.is_builtin else "否", d.id
        ])

    df = pd.DataFrame(data, columns=[
        "名称", "总图片数", "类别数", "上传时间", "预置", "ID"
    ])
    return df


def run_single_image_attack(image, model_id, attack_method, epsilon, norm,
                           targeted, target_class, **attack_params):
    if image is None:
        return None, "请上传图片"
    if not model_id:
        return None, "请选择模型"

    try:
        model_info = model_manager.get_model(model_id)
        model = model_manager.load_model(model_id)
        model.eval()

        input_size = model_info.input_size
        img_tensor = image_to_tensor(image, input_size=input_size, normalize=False).to(DEVICE)

        with torch.no_grad():
            clean_output = model(normalize_image(img_tensor))
            clean_prob = torch.softmax(clean_output, dim=1)
            clean_pred = clean_output.argmax(1).item()
            clean_conf = clean_prob[0, clean_pred].item()

        target_labels = None
        if targeted:
            target_labels = torch.tensor([target_class], device=DEVICE)

        attack_params_dict = {
            "epsilon": epsilon / 255.0,
            "norm": norm,
            "targeted": targeted,
            **attack_params
        }

        attack = get_attack_instance(attack_method, model, attack_params_dict)

        labels = torch.tensor([clean_pred], device=DEVICE)
        adv_tensor = attack.generate(img_tensor, labels, target_labels)

        with torch.no_grad():
            adv_output = model(normalize_image(adv_tensor))
            adv_prob = torch.softmax(adv_output, dim=1)
            adv_pred = adv_output.argmax(1).item()
            adv_conf = adv_prob[0, adv_pred].item()

        quality = quality_metrics.compute_all(img_tensor, adv_tensor)
        linf, l2 = compute_perturbation_metrics(img_tensor, adv_tensor)

        comparison_img = visualizer.create_adv_comparison(
            img_tensor, adv_tensor, clean_pred, adv_pred,
            clean_conf=clean_conf, adv_conf=adv_conf,
            labels=model_info.labels
        )

        success = adv_pred != clean_pred
        status = "✓ 攻击成功" if success else "✗ 攻击失败"

        result_text = f"""
        {status}
        原始预测: {model_info.labels[clean_pred] if clean_pred < len(model_info.labels) else f'class_{clean_pred}'} ({clean_pred})
        原始置信度: {clean_conf:.4f}
        对抗预测: {model_info.labels[adv_pred] if adv_pred < len(model_info.labels) else f'class_{adv_pred}'} ({adv_pred})
        对抗置信度: {adv_conf:.4f}

        扰动指标:
        L∞: {linf:.6f} ({linf * 255:.2f}/255)
        L2: {l2:.6f}

        质量指标:
        SSIM: {quality['ssim']:.4f} {'⚠️ 人眼可见' if quality['low_quality_warning'] else ''}
        PSNR: {quality['psnr']:.2f} dB
        LPIPS: {quality['lpips']:.4f}
        """

        return comparison_img, result_text
    except Exception as e:
        return None, f"攻击失败: {str(e)}"


def run_batch_evaluation(model_id, dataset_id, attack_method, epsilon, norm,
                         progress=gr.Progress(), **attack_params):
    if not model_id:
        return None, "请选择模型"
    if not dataset_id:
        return None, "请选择数据集"

    try:
        model_info = model_manager.get_model(model_id)
        input_size = model_info.input_size

        dataloader = dataset_manager.get_dataloader(
            dataset_id, batch_size=8, input_size=input_size, shuffle=False
        )

        attack_params_dict = {
            "epsilon": epsilon / 255.0,
            "norm": norm,
            **attack_params
        }

        def attack_fn(model, images, labels, **kwargs):
            attack = get_attack_instance(attack_method, model, kwargs)
            return attack.generate(images, labels)

        def progress_cb(current, total):
            progress(current / total, desc="评估中...")

        metrics = robustness_metrics.evaluate(
            model_id, dataloader, attack_fn, attack_params_dict, progress_cb
        )

        model_manager.evaluate_clean_accuracy(model_id, dataloader)

        experiment_manager.record_experiment(
            model_id, dataset_id, attack_method, attack_params_dict, metrics
        )

        acc_fig = visualizer.create_accuracy_bar_chart(
            metrics["clean_accuracy"], metrics["robust_accuracy"]
        )
        heatmap_fig = visualizer.create_per_class_heatmap(
            metrics["per_class_robustness"], model_info.labels
        )

        result_text = f"""
        评估完成！

        Clean Accuracy: {metrics['clean_accuracy']:.2f}%
        Robust Accuracy: {metrics['robust_accuracy']:.2f}%
        Attack Success Rate: {metrics['attack_success_rate']:.2f}%
        Average Perturbation (L2): {metrics['average_perturbation_l2']:.6f}
        Average Perturbation (L∞): {metrics['average_perturbation_linf']:.6f}
        Confidence Drop: {metrics['confidence_drop']:.2f}%

        总样本数: {metrics['total_samples']}
        Clean正确: {metrics['clean_correct']}
        Robust正确: {metrics['robust_correct']}
        攻击成功: {metrics['attack_success']}
        """

        return result_text, acc_fig, heatmap_fig
    except Exception as e:
        return f"评估失败: {str(e)}", None, None


def run_defense_comparison(model_id, defense_type, jpeg_q, min_s, max_s, bits, kernel,
                           model_id_defense, dataset_id, attack_method,
                           epsilon, norm, progress=gr.Progress()):
    if not model_id:
        return None, "请选择原始模型"
    if not dataset_id:
        return None, "请选择数据集"

    try:
        defense_params = {}
        if defense_type == "jpeg_compression":
            defense_params = {"quality": jpeg_q}
        elif defense_type == "random_resize_padding":
            defense_params = {"min_scale": min_s, "max_scale": max_s}
        elif defense_type == "bit_depth_reduction":
            defense_params = {"bits": bits}
        elif defense_type == "median_filter":
            defense_params = {"kernel_size": int(kernel)}

        model_info = model_manager.get_model(model_id)
        input_size = model_info.input_size

        dataloader = dataset_manager.get_dataloader(
            dataset_id, batch_size=8, input_size=input_size, shuffle=False
        )

        attack_params_dict = {
            "epsilon": epsilon / 255.0,
            "norm": norm,
        }

        def attack_fn(model, images, labels, **kwargs):
            attack = get_attack_instance(attack_method, model, kwargs)
            return attack.generate(images, labels)

        original_metrics = robustness_metrics.evaluate(
            model_id, dataloader, attack_fn, attack_params_dict
        )

        if defense_type != "adversarial_model":
            original_model = model_manager.load_model(model_id)
            defended_model = defense_methods.wrap_model_with_defense(
                original_model, defense_type, **defense_params
            )
            from src.metrics.robustness_metrics import RobustnessMetrics as RM
            metrics_calculator = RM(model_manager)

            def defended_attack_fn(att_model, images, labels, **kwargs):
                attack = get_attack_instance(attack_method, att_model, kwargs)
                return attack.generate(images, labels)

            defense_metrics = metrics_calculator.evaluate(
                model_id, dataloader, defended_attack_fn, attack_params_dict,
                model=defended_model,
                attack_model=original_model
            )
        else:
            if not model_id_defense:
                return None, "请选择对抗训练模型"
            defense_metrics = robustness_metrics.evaluate(
                model_id_defense, dataloader, attack_fn, attack_params_dict
            )

        comparison = {
            "original": original_metrics,
            "defense": defense_metrics,
            "improvements": {
                "clean_accuracy": defense_metrics["clean_accuracy"] - original_metrics["clean_accuracy"],
                "robust_accuracy": defense_metrics["robust_accuracy"] - original_metrics["robust_accuracy"],
                "attack_success_rate": original_metrics["attack_success_rate"] - defense_metrics["attack_success_rate"],
            }
        }

        fig = visualizer.create_defense_comparison_figure(original_metrics, defense_metrics)

        imp = comparison["improvements"]
        result_text = f"""
        原始模型:
        Clean Accuracy: {original_metrics['clean_accuracy']:.2f}%
        Robust Accuracy: {original_metrics['robust_accuracy']:.2f}%
        Attack Success Rate: {original_metrics['attack_success_rate']:.2f}%

        防御后模型:
        Clean Accuracy: {defense_metrics['clean_accuracy']:.2f}%
        Robust Accuracy: {defense_metrics['robust_accuracy']:.2f}%
        Attack Success Rate: {defense_metrics['attack_success_rate']:.2f}%

        变化:
        Clean Accuracy: {imp['clean_accuracy']:+.2f}% {'✅ 改善' if imp['clean_accuracy'] >= 0 else '❌ 恶化'}
        Robust Accuracy: {imp['robust_accuracy']:+.2f}% {'✅ 改善' if imp['robust_accuracy'] >= 0 else '❌ 恶化'}
        Attack Success Rate: {imp['attack_success_rate']:+.2f}% {'✅ 改善' if imp['attack_success_rate'] >= 0 else '❌ 恶化'}
        """

        return result_text, fig
    except Exception as e:
        return f"对比失败: {str(e)}", None


def run_transferability_analysis(model_ids, dataset_id, attack_method,
                                 epsilon, norm, progress=gr.Progress()):
    if len(model_ids) < 2:
        return None, "请至少选择2个模型"
    if not dataset_id:
        return None, "请选择数据集"

    try:
        analyzer = TransferabilityAnalyzer(model_manager)
        dataset_info = dataset_manager.get_dataset(dataset_id)
        model_info = model_manager.get_model(model_ids[0])
        input_size = model_info.input_size

        dataloader = dataset_manager.get_dataloader(
            dataset_id, batch_size=8, input_size=input_size, shuffle=False
        )

        attack_params_dict = {
            "epsilon": epsilon / 255.0,
            "norm": norm,
            "iterations": 20,
            "alpha": 2.0 / 255.0,
            "random_start": True,
        }

        def attack_fn(model, images, labels, **kwargs):
            attack = PGD(model, **kwargs)
            return attack.generate(images, labels)

        def progress_cb(current, total):
            progress(current / total, desc="计算迁移性矩阵...")

        matrix, model_names = analyzer.compute_transferability_matrix(
            model_ids, dataloader, attack_fn, attack_params_dict, progress_cb
        )

        fig = visualizer.create_transferability_heatmap(matrix, model_names)

        result_text = "迁移性矩阵计算完成！\n\n"
        result_text += "对角线为白盒攻击成功率，非对角线为迁移攻击成功率。\n"
        result_text += "高迁移率表示模型之间相似度高，适合集成防御。\n\n"

        for i, name in enumerate(model_names):
            avg_transfer = np.mean([matrix[i, j] for j in range(len(model_names)) if i != j])
            result_text += f"{name}: 平均迁移率 {avg_transfer * 100:.2f}%\n"

        return result_text, fig
    except Exception as e:
        return f"分析失败: {str(e)}", None


def get_experiment_records():
    experiments = experiment_manager.list_experiments()

    data = []
    for exp in experiments:
        metrics = exp["metrics"]
        model_info = model_manager.get_model(exp["model_id"])
        dataset_info = dataset_manager.get_dataset(exp["dataset_id"])
        model_name = model_info.name if model_info else exp["model_id"]
        dataset_name = dataset_info.name if dataset_info else exp["dataset_id"]

        data.append([
            exp["timestamp"], model_name, dataset_name,
            ATTACK_METHODS.get(exp["attack_method"], {}).get("name", exp["attack_method"]),
            f"{exp['attack_params'].get('epsilon', 0) * 255:.0f}/255",
            f"{metrics['clean_accuracy']:.2f}%",
            f"{metrics['robust_accuracy']:.2f}%",
            f"{metrics['attack_success_rate']:.2f}%",
            exp["id"]
        ])

    df = pd.DataFrame(data, columns=[
        "时间", "模型", "数据集", "攻击方法", "Epsilon",
        "Clean Acc", "Robust Acc", "Attack Success", "ID"
    ])
    return df


def batch_evaluation(model_id, dataset_id, attack_methods, epsilons,
                     progress=gr.Progress()):
    if not model_id or not dataset_id:
        return None, "请选择模型和数据集"
    if not attack_methods:
        return None, "请选择至少一种攻击方法"
    if not epsilons:
        return None, "请至少选择一个epsilon值"

    try:
        all_results = []
        total_tasks = len(attack_methods) * len(epsilons)
        current_task = 0

        model_info = model_manager.get_model(model_id)
        input_size = model_info.input_size

        dataloader = dataset_manager.get_dataloader(
            dataset_id, batch_size=8, input_size=input_size, shuffle=False
        )

        for attack_method in attack_methods:
            for eps in epsilons:
                current_task += 1
                progress(current_task / total_tasks,
                        desc=f"评估 {ATTACK_METHODS[attack_method]['name']} ε={eps}/255")

                attack_params_dict = {
                    "epsilon": eps / 255.0,
                    "norm": "Linf",
                    "iterations": 20,
                    "alpha": 2.0 / 255.0,
                    "random_start": True,
                }

                def attack_fn(model, images, labels, **kwargs):
                    attack = get_attack_instance(attack_method, model, kwargs)
                    return attack.generate(images, labels)

                metrics = robustness_metrics.evaluate(
                    model_id, dataloader, attack_fn, attack_params_dict
                )

                exp_id = experiment_manager.record_experiment(
                    model_id, dataset_id, attack_method, attack_params_dict, metrics
                )

                all_results.append({
                    "attack": ATTACK_METHODS[attack_method]['name'],
                    "epsilon": f"{eps}/255",
                    "clean_acc": metrics["clean_accuracy"],
                    "robust_acc": metrics["robust_accuracy"],
                    "attack_success": metrics["attack_success_rate"],
                    "exp_id": exp_id
                })

        df = pd.DataFrame(all_results)
        df.columns = ["攻击方法", "Epsilon", "Clean Acc (%)", "Robust Acc (%)", "Attack Success (%)", "实验ID"]

        fig = go.Figure()
        for attack in df["攻击方法"].unique():
            subset = df[df["攻击方法"] == attack]
            fig.add_trace(go.Scatter(
                x=subset["Epsilon"],
                y=subset["Robust Acc (%)"],
                mode='lines+markers',
                name=attack
            ))
        fig.update_layout(
            title="鲁棒准确率 vs Epsilon",
            xaxis_title="Epsilon",
            yaxis_title="Robust Accuracy (%)",
            yaxis_range=[0, 100]
        )

        return df, fig
    except Exception as e:
        return f"批量评估失败: {str(e)}", None


def get_attack_params_ui(attack_method):
    params = []
    if attack_method == "fgsm":
        params = []
    elif attack_method == "pgd":
        params = [
            gr.Slider(minimum=1, maximum=100, value=20, step=1, label="迭代次数"),
            gr.Slider(minimum=0.1, maximum=8.0, value=2.0, step=0.1, label="步长 alpha (/255)"),
            gr.Checkbox(value=True, label="随机起点"),
        ]
    elif attack_method == "cw":
        params = [
            gr.Slider(minimum=100, maximum=5000, value=1000, step=100, label="最大迭代次数"),
            gr.Slider(minimum=3, maximum=15, value=9, step=1, label="二分搜索步数"),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Kappa"),
        ]
    elif attack_method == "deepfool":
        params = [
            gr.Slider(minimum=10, maximum=200, value=50, step=5, label="最大迭代次数"),
            gr.Slider(minimum=1.0, maximum=1.2, value=1.02, step=0.01, label="过冲系数"),
        ]
    elif attack_method == "autoattack":
        params = [
            gr.CheckboxGroup(
                choices=["apgd_ce", "apgd_t", "fab", "square"],
                value=["apgd_ce", "apgd_t", "fab", "square"],
                label="攻击组件"
            ),
        ]
    elif attack_method == "square":
        params = [
            gr.Slider(minimum=1000, maximum=20000, value=5000, step=500, label="查询预算"),
            gr.Slider(minimum=0.1, maximum=0.9, value=0.8, step=0.1, label="初始方形大小 p"),
        ]
    return params


def create_app():
    with gr.Blocks(title="对抗样本生成与模型鲁棒性评估平台") as app:
        gr.Markdown("# 🛡️ 对抗样本生成与模型鲁棒性评估平台")
        gr.Markdown("系统性测试图像分类模型在对抗攻击下的表现，找出模型的脆弱点")

        with gr.Tabs():
            with gr.TabItem("🏠 首页"):
                gr.Markdown("## 功能概览")
                gr.Markdown("""
                - **模型管理**: 上传PyTorch/ONNX模型，自动检测输入尺寸和类别数
                - **数据集管理**: 上传测试图片集或使用预置ImageNet子集
                - **单张图片攻击**: 快速测试单个样本的对抗攻击效果
                - **批量评估**: 对整个测试集运行攻击，计算鲁棒性指标
                - **防御对比**: 测试输入变换防御和对抗训练模型的效果
                - **可迁移性分析**: 分析对抗样本在不同模型间的迁移能力
                - **实验记录**: 历史评估记录和对比分析
                """)

                gr.Markdown("## 支持的攻击方法")
                attack_info = pd.DataFrame([
                    ["FGSM", "白盒", "单步快速梯度符号法", "快"],
                    ["PGD", "白盒", "迭代投影梯度下降", "中"],
                    ["C&W", "白盒", "基于优化的最小扰动攻击", "慢"],
                    ["DeepFool", "白盒", "最小距离到决策边界", "中"],
                    ["AutoAttack", "白盒", "集成攻击（最可靠基准）", "慢"],
                    ["Square Attack", "黑盒", "无梯度随机搜索", "中"],
                ], columns=["攻击方法", "类型", "描述", "速度"])
                gr.Dataframe(attack_info, interactive=False)

            with gr.TabItem("📦 模型管理"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 上传模型")
                        model_file = gr.File(label="模型文件 (.pt, .pth, .onnx)")
                        model_name = gr.Textbox(label="模型名称")
                        model_arch = gr.Dropdown(
                            choices=SUPPORTED_ARCHITECTURES,
                            value="resnet50",
                            label="模型架构"
                        )
                        is_state_dict = gr.Checkbox(value=True, label="是state_dict格式")
                        labels_file = gr.File(label="类别标签文件（可选，每行一个类别名）")
                        upload_btn = gr.Button("上传模型", variant="primary")

                    with gr.Column():
                        gr.Markdown("### 模型列表")
                        model_list = gr.Dataframe(interactive=False, label="已上传模型")
                        refresh_btn = gr.Button("刷新列表")

                model_status = gr.Textbox(label="状态", interactive=False)

                upload_btn.click(
                    upload_model,
                    inputs=[model_file, model_name, model_arch, is_state_dict, labels_file],
                    outputs=[model_status, model_list]
                )
                refresh_btn.click(refresh_model_list, outputs=model_list)

                with gr.Row():
                    delete_model_id = gr.Textbox(label="要删除的模型ID")
                    delete_btn = gr.Button("删除模型", variant="stop")
                    delete_btn.click(delete_model, inputs=delete_model_id, outputs=[model_status, model_list])

            with gr.TabItem("📊 数据集管理"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 上传数据集")
                        dataset_file = gr.File(label="数据集ZIP文件（按类别文件夹组织）")
                        dataset_name = gr.Textbox(label="数据集名称")
                        upload_ds_btn = gr.Button("上传数据集", variant="primary")

                    with gr.Column():
                        gr.Markdown("### 数据集列表")
                        dataset_list = gr.Dataframe(interactive=False, label="已有数据集")
                        refresh_ds_btn = gr.Button("刷新列表")

                dataset_status = gr.Textbox(label="状态", interactive=False)
                gr.Markdown("### 预置数据集")
                gr.Info("已预置 ImageNet 验证子集（100张），可直接使用。")

                upload_ds_btn.click(
                    upload_dataset,
                    inputs=[dataset_file, dataset_name],
                    outputs=[dataset_status, dataset_list]
                )
                refresh_ds_btn.click(refresh_dataset_list, outputs=dataset_list)

            with gr.TabItem("🎯 单张图片攻击"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(type="pil", label="输入图片")
                        model_select_single = gr.Dropdown(
                            choices=get_model_choices(),
                            label="选择模型"
                        )
                        attack_select_single = gr.Dropdown(
                            choices=[(v["name"], k) for k, v in ATTACK_METHODS.items()],
                            value="pgd",
                            label="攻击方法"
                        )

                        with gr.Row():
                            epsilon_single = gr.Slider(
                                minimum=0, maximum=32, value=8, step=1,
                                label="扰动预算 epsilon (/255)"
                            )
                            norm_single = gr.Radio(
                                choices=["Linf", "L2"], value="Linf",
                                label="范数类型"
                            )

                        targeted_single = gr.Checkbox(value=False, label="Targeted 攻击")
                        target_class_single = gr.Number(
                            value=0, precision=0, label="目标类别", visible=False
                        )
                        targeted_single.change(
                            lambda x: gr.update(visible=x),
                            inputs=targeted_single, outputs=target_class_single
                        )

                        attack_params_container = gr.Column()
                        attack_select_single.change(
                            get_attack_params_ui,
                            inputs=attack_select_single,
                            outputs=attack_params_container
                        )

                        run_single_btn = gr.Button("生成对抗样本", variant="primary")

                    with gr.Column():
                        result_image = gr.Image(label="结果对比")
                        result_text_single = gr.Textbox(label="结果", interactive=False, lines=15)

                run_single_btn.click(
                    run_single_image_attack,
                    inputs=[
                        input_image, model_select_single, attack_select_single,
                        epsilon_single, norm_single, targeted_single, target_class_single
                    ],
                    outputs=[result_image, result_text_single]
                )

                gr.Markdown("### 常用参数预设")
                preset_choices = [(v["name"], k) for k, v in preset_manager.list_presets().items()]
                preset_select = gr.Dropdown(choices=preset_choices, label="加载预设")

                def load_preset(preset_id):
                    preset = preset_manager.get_preset(preset_id)
                    if not preset:
                        return {}
                    params = preset["params"]
                    return [
                        params.get("epsilon", 8) * 255,
                        params.get("norm", "Linf"),
                        params.get("targeted", False),
                    ]

                preset_select.change(
                    load_preset,
                    inputs=preset_select,
                    outputs=[epsilon_single, norm_single, targeted_single]
                )

            with gr.TabItem("📈 批量评估"):
                with gr.Row():
                    with gr.Column():
                        model_select_batch = gr.Dropdown(
                            choices=get_model_choices(), label="选择模型"
                        )
                        dataset_select_batch = gr.Dropdown(
                            choices=get_dataset_choices(), label="选择数据集"
                        )
                        attack_select_batch = gr.Dropdown(
                            choices=[(v["name"], k) for k, v in ATTACK_METHODS.items()],
                            value="pgd", label="攻击方法"
                        )
                        epsilon_batch = gr.Slider(
                            minimum=0, maximum=32, value=8, step=1,
                            label="扰动预算 epsilon (/255)"
                        )
                        norm_batch = gr.Radio(
                            choices=["Linf", "L2"], value="Linf", label="范数类型"
                        )
                        run_batch_btn = gr.Button("开始评估", variant="primary")

                    with gr.Column():
                        result_text_batch = gr.Textbox(label="评估结果", interactive=False, lines=12)
                        acc_chart = gr.Plot(label="准确率对比")
                        heatmap_chart = gr.Plot(label="各类别鲁棒性")

                run_batch_btn.click(
                    run_batch_evaluation,
                    inputs=[
                        model_select_batch, dataset_select_batch, attack_select_batch,
                        epsilon_batch, norm_batch
                    ],
                    outputs=[result_text_batch, acc_chart, heatmap_chart]
                )

            with gr.TabItem("🛡️ 防御对比"):
                with gr.Row():
                    with gr.Column():
                        model_select_def = gr.Dropdown(
                            choices=get_model_choices(), label="原始模型"
                        )
                        defense_type = gr.Radio(
                            choices=[
                                ("JPEG压缩", "jpeg_compression"),
                                ("随机缩放+填充", "random_resize_padding"),
                                ("位深度缩减", "bit_depth_reduction"),
                                ("中值滤波", "median_filter"),
                                ("对抗训练模型", "adversarial_model"),
                            ],
                            value="jpeg_compression",
                            label="防御方法"
                        )

                        with gr.Column(visible=True) as jpeg_params:
                            jpeg_quality = gr.Slider(1, 100, value=50, label="JPEG质量")
                        with gr.Column(visible=False) as resize_params:
                            min_scale = gr.Slider(0.5, 1.0, value=0.8, label="最小缩放")
                            max_scale = gr.Slider(1.0, 1.5, value=1.2, label="最大缩放")
                        with gr.Column(visible=False) as bit_params:
                            bit_depth = gr.Slider(1, 8, value=4, step=1, label="位深度")
                        with gr.Column(visible=False) as median_params:
                            kernel_size = gr.Slider(3, 7, value=3, step=2, label="核大小")
                        with gr.Column(visible=False) as adv_model_params:
                            model_select_def2 = gr.Dropdown(
                                choices=get_model_choices(), label="对抗训练模型"
                            )

                        def update_defense_params(def_type):
                            return [
                                gr.update(visible=def_type == "jpeg_compression"),
                                gr.update(visible=def_type == "random_resize_padding"),
                                gr.update(visible=def_type == "bit_depth_reduction"),
                                gr.update(visible=def_type == "median_filter"),
                                gr.update(visible=def_type == "adversarial_model"),
                            ]

                        defense_type.change(
                            update_defense_params,
                            inputs=defense_type,
                            outputs=[jpeg_params, resize_params, bit_params, median_params, adv_model_params]
                        )

                        dataset_select_def = gr.Dropdown(
                            choices=get_dataset_choices(), label="选择数据集"
                        )
                        attack_select_def = gr.Dropdown(
                            choices=[(v["name"], k) for k, v in ATTACK_METHODS.items()],
                            value="pgd", label="攻击方法"
                        )
                        epsilon_def = gr.Slider(
                            minimum=0, maximum=32, value=8, step=1,
                            label="扰动预算 epsilon (/255)"
                        )
                        norm_def = gr.Radio(
                            choices=["Linf", "L2"], value="Linf", label="范数类型"
                        )
                        run_defense_btn = gr.Button("开始对比评估", variant="primary")

                    with gr.Column():
                        result_text_def = gr.Textbox(label="对比结果", interactive=False, lines=18)
                        defense_chart = gr.Plot(label="防御效果对比")

                def get_defense_params(def_type, jpeg_q, min_s, max_s, bits, kernel):
                    if def_type == "jpeg_compression":
                        return {"quality": jpeg_q}
                    elif def_type == "random_resize_padding":
                        return {"min_scale": min_s, "max_scale": max_s}
                    elif def_type == "bit_depth_reduction":
                        return {"bits": bits}
                    elif def_type == "median_filter":
                        return {"kernel_size": kernel}
                    return {}

                run_defense_btn.click(
                    run_defense_comparison,
                    inputs=[
                        model_select_def, defense_type,
                        jpeg_quality, min_scale, max_scale, bit_depth, kernel_size,
                        model_select_def2, dataset_select_def,
                        attack_select_def, epsilon_def, norm_def
                    ],
                    outputs=[result_text_def, defense_chart]
                )

            with gr.TabItem("🔄 可迁移性分析"):
                with gr.Row():
                    with gr.Column():
                        model_select_transfer = gr.CheckboxGroup(
                            choices=get_model_choices(), label="选择模型（至少2个）"
                        )
                        dataset_select_transfer = gr.Dropdown(
                            choices=get_dataset_choices(), label="选择数据集"
                        )
                        attack_select_transfer = gr.Dropdown(
                            choices=[(v["name"], k) for k, v in ATTACK_METHODS.items()],
                            value="pgd", label="攻击方法"
                        )
                        epsilon_transfer = gr.Slider(
                            minimum=0, maximum=32, value=8, step=1,
                            label="扰动预算 epsilon (/255)"
                        )
                        norm_transfer = gr.Radio(
                            choices=["Linf", "L2"], value="Linf", label="范数类型"
                        )
                        run_transfer_btn = gr.Button("分析可迁移性", variant="primary")

                    with gr.Column():
                        result_text_transfer = gr.Textbox(label="分析结果", interactive=False, lines=10)
                        transfer_heatmap = gr.Plot(label="迁移性矩阵")

                run_transfer_btn.click(
                    run_transferability_analysis,
                    inputs=[
                        model_select_transfer, dataset_select_transfer,
                        attack_select_transfer, epsilon_transfer, norm_transfer
                    ],
                    outputs=[result_text_transfer, transfer_heatmap]
                )

            with gr.TabItem("⚡ 批量评估模式"):
                with gr.Row():
                    with gr.Column():
                        model_select_multi = gr.Dropdown(
                            choices=get_model_choices(), label="选择模型"
                        )
                        dataset_select_multi = gr.Dropdown(
                            choices=get_dataset_choices(), label="选择数据集"
                        )
                        attacks_multi = gr.CheckboxGroup(
                            choices=[(v["name"], k) for k, v in ATTACK_METHODS.items()],
                            value=["fgsm", "pgd"],
                            label="攻击方法（可多选）"
                        )
                        epsilons_multi = gr.CheckboxGroup(
                            choices=[2, 4, 8, 16, 32], value=[2, 4, 8, 16],
                            label="Epsilon 值 (/255)（可多选）"
                        )
                        run_multi_btn = gr.Button("一键全量评估", variant="primary")

                    with gr.Column():
                        result_table_multi = gr.Dataframe(label="评估结果")
                        result_chart_multi = gr.Plot(label="鲁棒准确率曲线")

                run_multi_btn.click(
                    batch_evaluation,
                    inputs=[
                        model_select_multi, dataset_select_multi,
                        attacks_multi, epsilons_multi
                    ],
                    outputs=[result_table_multi, result_chart_multi]
                )

            with gr.TabItem("📋 实验记录"):
                with gr.Row():
                    with gr.Column():
                        refresh_exp_btn = gr.Button("刷新记录")
                        experiment_table = gr.Dataframe(interactive=False, label="历史实验记录")

                    with gr.Column():
                        compare_ids = gr.Textbox(label="对比实验ID（用逗号分隔）")
                        compare_btn = gr.Button("对比选中实验")
                        compare_result = gr.Plot(label="对比图表")

                refresh_exp_btn.click(get_experiment_records, outputs=experiment_table)

                def compare_experiments(ids_str):
                    ids = [s.strip() for s in ids_str.split(",") if s.strip()]
                    experiments = experiment_manager.compare_experiments(ids)
                    if not experiments:
                        return None

                    metrics_list = [exp["metrics"] for exp in experiments]
                    names = [f"实验{exp['id']}" for exp in experiments]
                    return visualizer.create_metrics_comparison_table(metrics_list, names)

                compare_btn.click(compare_experiments, inputs=compare_ids, outputs=compare_result)

        app.load(refresh_model_list, outputs=model_list)
        app.load(refresh_dataset_list, outputs=dataset_list)
        app.load(get_experiment_records, outputs=experiment_table)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
