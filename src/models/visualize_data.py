
# scripts/visualize_data.py
"""
该脚本用于可视化卫星轨道根数随时间的变化，
并在图上标记出已知的机动事件，以便进行直观分析。
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 路径设置 ---
# 确保能够导入src目录下的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.data.loader import SatelliteDataLoader
except ImportError as e:
    print(f"❌ 无法导入模块，请确保您的项目结构正确: {e}")
    sys.exit(1)


def visualize_parameters_with_maneuvers(tle_data: pd.DataFrame, maneuver_times: list, satellite_name: str):
    """
    可视化指定参数随时间的变化，并标记机动事件。

    Args:
        tle_data (pd.DataFrame): 包含轨道根数和'epoch'列的数据。
        maneuver_times (list): 包含机动事件时间戳的列表。
        satellite_name (str): 卫星名称，用于图表标题。
    """
    print(f"🎨 正在为卫星 {satellite_name} 生成可视化图表...")

    # 要可视化的参数列表
    params_to_plot = ['mean_motion', 'eccentricity', 'inclination']
    
    # 确保数据以时间为索引
    if 'epoch' in tle_data.columns:
        data = tle_data.set_index('epoch').sort_index()
    else:
        print("❌ 数据中未找到 'epoch' 列。")
        return

    # 设置图表布局：3个子图，共享X轴
    fig, axes = plt.subplots(len(params_to_plot), 1, figsize=(18, 12), sharex=True)
    fig.suptitle(f'Orbital Parameters of {satellite_name} with Maneuver Events', fontsize=18, y=0.96)

    # 循环绘制每个参数
    for i, param in enumerate(params_to_plot):
        ax = axes[i]
        
        # 绘制参数曲线
        ax.plot(data.index, data[param], marker='.', linestyle='-', markersize=3, label=param)
        
        # 标记机动事件
        for j, maneuver_time in enumerate(maneuver_times):
            # 只在第一个子图的第一个标记上添加图例，避免重复
            label = 'Maneuver Event' if i == 0 and j == 0 else '_nolegend_'
            ax.axvline(x=maneuver_time, color='red', linestyle='--', linewidth=1.2, label=label)

        # 设置子图样式
        ax.set_ylabel(param, fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')

    # 设置X轴样式
    axes[-1].set_xlabel('Date', fontsize=12)
    # 优化X轴日期的显示格式
    fig.autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # 每3个月一个主刻度
    
    # 调整布局并显示图表
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    print("✅ 图表生成完成，正在显示...")
    plt.show()


def main():
    """主执行函数"""
    SATELLITE_NAME = "Fengyun-4A"
    
    print("STEP 1: 加载数据...")
    try:
        loader = SatelliteDataLoader(data_dir="data")
        tle_data, maneuver_times = loader.load_satellite_data(SATELLITE_NAME)
        if tle_data is None:
            print("数据加载失败，任务终止。")
            return
            
        print(f"✅ 数据加载成功: {len(tle_data)} 条记录, {len(maneuver_times)} 次机动。")
        
        visualize_parameters_with_maneuvers(tle_data, maneuver_times, SATELLITE_NAME)
        
    except Exception as e:
        print(f"❌ 执行过程中发生错误: {e}")


if __name__ == "__main__":
    main()