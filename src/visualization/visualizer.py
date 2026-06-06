from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont

from src.utils.helpers import tensor_to_image, get_imagenet_labels


class Visualizer:
    def __init__(self):
        sns.set_style("whitegrid")

    def create_adv_comparison(self, original: torch.Tensor, adversarial: torch.Tensor,
                              clean_pred: int, adv_pred: int, true_label: Optional[int] = None,
                              clean_conf: float = 0.0, adv_conf: float = 0.0,
                              labels: Optional[List[str]] = None) -> Image.Image:
        if labels is None:
            labels = get_imagenet_labels()

        orig_img = tensor_to_image(original, denormalize=False)
        adv_img = tensor_to_image(adversarial, denormalize=False)

        perturbation = (adversarial - original).squeeze(0).detach()
        pert_np = perturbation.permute(1, 2, 0).cpu().numpy()
        pert_np = (pert_np - pert_np.min()) / (pert_np.max() - pert_np.min() + 1e-8)
        pert_img = (pert_np * 255).astype(np.uint8)

        linf = torch.max(torch.abs(perturbation)).item()
        l2 = torch.norm(perturbation.view(-1), p=2).item()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(orig_img)
        clean_label_name = labels[clean_pred] if clean_pred < len(labels) else f"class_{clean_pred}"
        title = f"原图\n预测: {clean_label_name} ({clean_pred})\n置信度: {clean_conf:.2f}"
        if true_label is not None:
            true_name = labels[true_label] if true_label < len(labels) else f"class_{true_label}"
            title += f"\n真实: {true_name} ({true_label})"
        axes[0].set_title(title, fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(pert_img)
        axes[1].set_title(f"扰动 (放大)\nL∞: {linf:.4f}\nL2: {l2:.4f}", fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(adv_img)
        adv_label_name = labels[adv_pred] if adv_pred < len(labels) else f"class_{adv_pred}"
        success = adv_pred != clean_pred
        status = "✓ 攻击成功" if success else "✗ 攻击失败"
        color = "red" if success else "green"
        title = f"对抗样本\n预测: {adv_label_name} ({adv_pred})\n置信度: {adv_conf:.2f}\n{status}"
        axes[2].set_title(title, fontsize=10, color=color)
        axes[2].axis('off')

        plt.tight_layout()

        buf = plt.savefig_to_buffer() if hasattr(plt, 'savefig_to_buffer') else None
        if buf is None:
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)

        comparison_img = Image.open(buf)
        plt.close(fig)
        return comparison_img

    def create_accuracy_bar_chart(self, clean_acc: float, robust_acc: float) -> go.Figure:
        fig = go.Figure(data=[
            go.Bar(name='Clean Accuracy', x=['Clean'], y=[clean_acc],
                   marker_color='rgb(55, 83, 109)'),
            go.Bar(name='Robust Accuracy', x=['Robust'], y=[robust_acc],
                   marker_color='rgb(26, 118, 255)')
        ])
        fig.update_layout(
            title='Clean vs Robust Accuracy',
            yaxis_title='Accuracy (%)',
            barmode='group',
            yaxis_range=[0, 100]
        )
        return fig

    def create_per_class_heatmap(self, per_class_robust: Dict[int, float],
                                 labels: Optional[List[str]] = None) -> go.Figure:
        if labels is None:
            labels = get_imagenet_labels()

        classes = sorted(per_class_robust.keys())
        values = [per_class_robust[c] for c in classes]
        class_names = [labels[c] if c < len(labels) else f"class_{c}" for c in classes]

        n_cols = min(10, len(classes))
        n_rows = (len(classes) + n_cols - 1) // n_cols

        values_matrix = np.zeros((n_rows, n_cols))
        labels_matrix = np.empty((n_rows, n_cols), dtype=object)

        for i, (cls, val, name) in enumerate(zip(classes, values, class_names)):
            row = i // n_cols
            col = i % n_cols
            values_matrix[row, col] = val
            labels_matrix[row, col] = f"{name}<br>{val:.1f}%"

        fig = px.imshow(
            values_matrix,
            text_auto=False,
            color_continuous_scale='RdYlGn',
            range_color=[0, 100],
            title='Per-Class Robustness (%)'
        )

        fig.update_traces(text=labels_matrix, texttemplate="%{text}")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        return fig

    def create_perturbation_histogram(self, perturbations: List[float],
                                      title: str = "Perturbation Distribution") -> go.Figure:
        fig = go.Figure(data=[go.Histogram(x=perturbations, nbinsx=30)])
        fig.update_layout(
            title=title,
            xaxis_title='Perturbation (L2)',
            yaxis_title='Count',
            bargap=0.2
        )
        return fig

    def create_transferability_heatmap(self, matrix: np.ndarray,
                                       model_names: List[str]) -> go.Figure:
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=model_names,
            y=model_names,
            hoverongaps=False,
            colorscale='Viridis',
            text=[[f"{v:.1f}%" for v in row] for row in matrix * 100],
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title='Transferability Matrix (%)',
            xaxis_title='Target Model',
            yaxis_title='Source Model',
        )
        return fig

    def create_metrics_comparison_table(self, metrics_list: List[Dict[str, Any]],
                                       names: List[str]) -> go.Figure:
        metric_keys = ["clean_accuracy", "robust_accuracy", "attack_success_rate",
                       "average_perturbation_l2"]
        metric_names = ["Clean Acc", "Robust Acc", "Attack Success", "Avg Pert (L2)"]

        values = []
        for metrics in metrics_list:
            row = []
            for key in metric_keys:
                val = metrics.get(key, 0)
                if "accuracy" in key or "success" in key:
                    row.append(f"{val:.2f}%")
                else:
                    row.append(f"{val:.4f}")
            values.append(row)

        fig = go.Figure(data=[go.Table(
            header=dict(values=["Model"] + metric_names,
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[names] + list(map(list, zip(*values))),
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(title="Metrics Comparison")
        return fig

    def create_defense_comparison_figure(self, original_metrics: Dict[str, Any],
                                         defense_metrics: Dict[str, Any]) -> go.Figure:
        keys = ["clean_accuracy", "robust_accuracy", "attack_success_rate"]
        names = ["Clean Accuracy", "Robust Accuracy", "Attack Success Rate"]

        orig_vals = [original_metrics.get(k, 0) for k in keys]
        def_vals = [defense_metrics.get(k, 0) for k in keys]
        improvements = [def_vals[i] - orig_vals[i] if "success" not in keys[i]
                        else orig_vals[i] - def_vals[i] for i in range(len(keys))]

        colors = ['green' if imp >= 0 else 'red' for imp in improvements]

        fig = go.Figure(data=[
            go.Bar(name='Original', x=names, y=orig_vals, marker_color='rgb(107, 107, 107)'),
            go.Bar(name='After Defense', x=names, y=def_vals, marker_color=colors)
        ])
        fig.update_layout(
            title='Defense Effect Comparison',
            yaxis_title='Percentage (%)',
            barmode='group'
        )
        return fig

    def create_thumbnail_grid(self, results: List[Dict[str, Any]],
                              n_cols: int = 5) -> Image.Image:
        n_rows = (len(results) + n_cols - 1) // n_cols
        thumb_size = 128
        margin = 5
        total_w = n_cols * (thumb_size + margin) + margin
        total_h = n_rows * (thumb_size + margin) + margin

        grid = Image.new('RGB', (total_w, total_h), color='white')

        for i, result in enumerate(results):
            row = i // n_cols
            col = i % n_cols
            x = margin + col * (thumb_size + margin)
            y = margin + row * (thumb_size + margin)

            img = tensor_to_image(result['adversarial'], denormalize=False)
            img_pil = Image.fromarray(img).resize((thumb_size, thumb_size))

            draw = ImageDraw.Draw(img_pil)
            success = result.get('success', False)
            border_color = 'red' if success else 'green'
            border_width = 3
            draw.rectangle([(0, 0), (thumb_size-1, thumb_size-1)],
                         outline=border_color, width=border_width)

            grid.paste(img_pil, (x, y))

        return grid

    def create_attack_comparison_radar(self, results: List[Dict[str, Any]]) -> go.Figure:
        categories = ['攻击成功率', '平均扰动量', '平均耗时', 'SSIM']

        fig = go.Figure()

        for result in results:
            attack_name = result['attack_name']
            success_rate = result['attack_success_rate'] / 100.0
            perturbation = min(result['average_perturbation_l2'] / 0.1, 1.0)
            time_norm = min(result['avg_time_per_sample'] / 1.0, 1.0)
            ssim_val = max(0, min(result.get('ssim_mean', 0), 1.0))

            values = [success_rate, perturbation, time_norm, ssim_val]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=attack_name,
                opacity=0.6
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            title=dict(
                text='攻击方法综合对比雷达图',
                x=0.5,
                xanchor='center'
            ),
            height=500
        )

        return fig

    def create_attack_comparison_table(self, results: List[Dict[str, Any]]) -> go.Figure:
        headers = ['攻击方法', 'Robust Accuracy (%)', 'Attack Success Rate (%)',
                   'Avg Perturbation (L2)', '平均耗时 (秒/样本)', 'SSIM均值']

        cell_values = [[], [], [], [], [], []]
        fill_colors = [[], [], [], [], [], []]

        cols_data = {
            1: 'robust_accuracy',
            2: 'attack_success_rate',
            3: 'average_perturbation_l2',
            4: 'avg_time_per_sample',
            5: 'ssim_mean'
        }

        higher_is_better = {1: True, 2: False, 3: False, 4: False, 5: True}

        for result in results:
            cell_values[0].append(result['attack_name'])
            fill_colors[0].append('white')

        for col_idx, metric_key in cols_data.items():
            values = [r.get(metric_key, 0) for r in results]

            if higher_is_better[col_idx]:
                best_val = max(values)
                worst_val = min(values)
            else:
                best_val = min(values)
                worst_val = max(values)

            for i, val in enumerate(values):
                if col_idx in [1, 2]:
                    cell_values[col_idx].append(f'{val:.2f}%')
                elif col_idx == 3:
                    cell_values[col_idx].append(f'{val:.6f}')
                elif col_idx == 4:
                    cell_values[col_idx].append(f'{val:.4f}')
                else:
                    cell_values[col_idx].append(f'{val:.4f}')

                if len(results) > 1 and val == best_val:
                    fill_colors[col_idx].append('#d4edda')
                elif len(results) > 1 and val == worst_val:
                    fill_colors[col_idx].append('#f8d7da')
                else:
                    fill_colors[col_idx].append('white')

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='#4472c4',
                font=dict(color='white', size=12),
                align='center',
                height=35
            ),
            cells=dict(
                values=cell_values,
                fill_color=fill_colors,
                align='center',
                font=dict(size=11),
                height=30
            )
        )])

        fig.update_layout(
            title=dict(
                text='攻击方法对比结果',
                x=0.5,
                xanchor='center'
            ),
            height=300 + len(results) * 35
        )

        return fig

    def create_training_loss_curve(self, epochs: List[int], losses: List[float],
                                   task_name: str = "Training") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs,
            y=losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='rgb(55, 83, 109)', width=2),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text=f'{task_name} - Loss Curve',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Epoch',
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            yaxis=dict(
                title='Average Loss'
            ),
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=50, r=50, t=60, b=50)
        )

        return fig

    def create_training_accuracy_curve(self, epochs: List[int],
                                       clean_accs: List[float],
                                       robust_accs: List[float],
                                       task_name: str = "Training") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs,
            y=clean_accs,
            mode='lines+markers',
            name='Clean Accuracy',
            line=dict(color='rgb(26, 118, 255)', width=2),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='Epoch %{x}<br>Clean Acc: %{y:.2f}%<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=epochs,
            y=robust_accs,
            mode='lines+markers',
            name='Robust Accuracy',
            line=dict(color='rgb(255, 65, 54)', width=2),
            marker=dict(size=8, symbol='square'),
            hovertemplate='Epoch %{x}<br>Robust Acc: %{y:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text=f'{task_name} - Accuracy Curve',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Epoch',
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            yaxis=dict(
                title='Accuracy (%)',
                range=[0, 100]
            ),
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=50, r=50, t=60, b=50)
        )

        return fig

    def create_training_comparison_charts(self,
                                          history1: Dict[str, Any],
                                          history2: Dict[str, Any]) -> go.Figure:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Loss Curve Comparison', 'Accuracy Curve Comparison'),
                            horizontal_spacing=0.15)

        name1 = f"{history1['task_info']['model_name']} ({history1['task_info']['id'][:8]})"
        name2 = f"{history2['task_info']['model_name']} ({history2['task_info']['id'][:8]})"

        fig.add_trace(go.Scatter(
            x=history1['epochs'],
            y=history1['losses'],
            mode='lines+markers',
            name=f'{name1} - Loss',
            line=dict(color='rgb(55, 83, 109)', width=2),
            marker=dict(size=6, symbol='circle'),
            legendgroup='loss'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=history2['epochs'],
            y=history2['losses'],
            mode='lines+markers',
            name=f'{name2} - Loss',
            line=dict(color='rgb(255, 127, 14)', width=2, dash='dash'),
            marker=dict(size=6, symbol='square'),
            legendgroup='loss'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=history1['epochs'],
            y=history1['clean_accs'],
            mode='lines+markers',
            name=f'{name1} - Clean Acc',
            line=dict(color='rgb(26, 118, 255)', width=2),
            marker=dict(size=6, symbol='circle'),
            legendgroup='acc'
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=history1['epochs'],
            y=history1['robust_accs'],
            mode='lines+markers',
            name=f'{name1} - Robust Acc',
            line=dict(color='rgb(255, 65, 54)', width=2),
            marker=dict(size=6, symbol='circle'),
            legendgroup='acc'
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=history2['epochs'],
            y=history2['clean_accs'],
            mode='lines+markers',
            name=f'{name2} - Clean Acc',
            line=dict(color='rgb(26, 118, 255)', width=2, dash='dash'),
            marker=dict(size=6, symbol='square'),
            legendgroup='acc'
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=history2['epochs'],
            y=history2['robust_accs'],
            mode='lines+markers',
            name=f'{name2} - Robust Acc',
            line=dict(color='rgb(255, 65, 54)', width=2, dash='dash'),
            marker=dict(size=6, symbol='square'),
            legendgroup='acc'
        ), row=1, col=2)

        fig.update_xaxes(title_text='Epoch', row=1, col=1, tickmode='linear', tick0=1, dtick=1)
        fig.update_xaxes(title_text='Epoch', row=1, col=2, tickmode='linear', tick0=1, dtick=1)
        fig.update_yaxes(title_text='Average Loss', row=1, col=1)
        fig.update_yaxes(title_text='Accuracy (%)', row=1, col=2, range=[0, 100])

        fig.update_layout(
            title=dict(
                text='Training History Comparison',
                x=0.5,
                xanchor='center'
            ),
            height=500,
            width=1200,
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5
            )
        )

        return fig
