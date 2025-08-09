# src/evaluation/reporting.py

import os

def save_detection_report(metrics, config, predictions, test_data_len):
    """
    一个通用的、模块化的报告生成函数。
    
    Args:
        metrics (dict): 包含精确率、召回率等性能指标的字典。
        config (dict): 包含卫星名称、输出路径等信息的全局配置字典。
        predictions (list): 最终预测出的异常时间戳列表。
        test_data_len (int): 测试集的长度。
    """
    print("\n📄 生成报告...")
    output_dir = config.get('output_dir', 'outputs/default')
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'detection_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{config['satellite_name']} 机动检测报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 配置概况:\n")
        f.write(f"   - 卫星名称: {config['satellite_name']}\n")
        f.write(f"   - 残差阈值乘数: {config['threshold_std_multiplier']}\n")
        f.write(f"   - 压力辅助触发阈值: {config['pressure_activation_threshold']}\n")
        f.write(f"   - 压力辅助折扣系数: {config['pressure_reduction_factor']}\n")
        f.write(f"   - 事件最小持续天数: {config['cluster_min_group_size']}\n\n")

        f.write("2. 检测结果:\n")
        pred_count = len(predictions) if predictions else 0
        f.write(f"   - 检测异常数: {pred_count}\n")
        f.write(f"   - 异常比例: {pred_count / test_data_len * 100:.2f}%\n\n")
        
        if metrics:
            f.write("3. 性能指标:\n")
            f.write(f"   - 精确率: {metrics['precision']:.3f}\n")
            f.write(f"   - 召回率: {metrics['recall']:.3f}\n")
            f.write(f"   - F1分数: {metrics['f1']:.3f}\n")
            f.write(f"   - TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}\n\n")
            
        f.write("4. 检测时间列表 (前20个):\n")
        if predictions:
            for i, pred_time in enumerate(predictions[:20]):
                f.write(f"   {i+1:2d}. {pred_time}\n")
            if len(predictions) > 20:
                f.write(f"   ... (还有 {len(predictions)-20} 个)\n")
        else:
            f.write("   无检测结果\n")
    
    print(f"   报告已保存到: {report_path}")