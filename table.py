import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# 定义所有模型名称，文件名格式为 test_results_{model_name}.json
model_names = [
    'agent_noise_fee_1',
    'agent_noise_fee_3',
    'agent_noise_fee_5',
    'agent_noise_fee_100',
    'agent_wtnoise'
]

# 加载各模型的测试结果
results = {}
for model in model_names:
    filename = f"test_results_{model}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            results[model] = json.load(f)
    else:
        print(f"File {filename} not found.")

# 假设每个测试结果文件中键为 "day0", "day1", ... 表示不同天的测试指标
if results:
    sample_model = next(iter(results.values()))
    # 对天数键进行排序（例如：day0, day1, ...）
    days = sorted(sample_model.keys(), key=lambda x: int(x.replace("day", "")))
else:
    days = []

# 将天数转换为数字列表
day_numbers = [int(day.replace("day", "")) for day in days]

# 指标列表
metrics = ["meanReward", "meanEff", "varEff", "overdueRate"]

# 创建 PDF 文件保存所有图表
output_pdf = "test_results_comparison.pdf"
with PdfPages(output_pdf) as pdf:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    # 对每个指标绘制一张图
    for i, metric in enumerate(metrics):
        ax = axs[i]
        for model in model_names:
            if model in results:
                values = [results[model][day][metric] for day in days]
                # 根据模型类型设置不同的画图风格
                if model == "agent_wtnoise":
                    # 原始模型风格：蓝色, 圆点, 实线
                    ax.plot(day_numbers, values, marker='o', linestyle='-', color='b', label=model)
                else:
                    # 带噪模型风格：正方形, 虚线, 半透明
                    ax.plot(day_numbers, values, marker='s', linestyle='--', alpha=0.7, label=model)
        ax.set_xlabel("Days")
        ax.set_ylabel(metric)
        ax.set_title(f"Trend of {metric} Over Days")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(day_numbers)
        ax.set_xticklabels(days, rotation=45)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"Comparison plots saved to {output_pdf}")
