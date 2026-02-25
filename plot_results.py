import re
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np 

def plot_log_data(log_file):
    """
    讀取日誌，列出所有 GA 觸發點，並將報告（含顏色解釋）寫入 TXT 檔案。
    """
    steps, rewards, drops = [], [], []
    ga_jam_periods, ga_penalty_periods = [], []
    is_test_mode = False 
    
    pending_jam, pending_penalty, pending_end = False, False, False
    jam_start, penalty_start = None, None
    current_step = 0

    pattern = re.compile(r"時間:\s*(\d+)s \| 綠燈(?:已亮)?:\s*\d+s \| 5秒獎勵:\s*(-?\d+\.\d+) \| 掉分:\s*(\d+)/20")

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "[TEST]" in line:
                    is_test_mode = True

                match = pattern.search(line)
                if match:
                    current_step = int(match.group(1))
                    steps.append(current_step)
                    rewards.append(float(match.group(2)))
                    drops.append(int(match.group(3))) 
                    
                    if pending_end:
                        if jam_start is not None:
                            ga_jam_periods.append((jam_start, current_step))
                            jam_start = None
                        if penalty_start is not None:
                            ga_penalty_periods.append((penalty_start, current_step))
                            penalty_start = None
                        pending_end = False

                    if pending_jam:
                        jam_start = current_step
                        pending_jam = False
                    if pending_penalty:
                        penalty_start = current_step
                        pending_penalty = False
                
                if "下游癱瘓" in line:
                    pending_jam = True
                elif "觸發 GA 保護機制" in line: 
                    pending_penalty = True
                elif "GA 示範結束" in line:
                    pending_end = True
                    
        if jam_start is not None: ga_jam_periods.append((jam_start, current_step))
        if penalty_start is not None: ga_penalty_periods.append((penalty_start, current_step))

        if not steps:
            print(f"❌ 找不到數據，請檢查檔名或格式。")
            return

        # ==========================================
        # 建立報告文字內容（含顏色解釋與模式標註）
        # ==========================================
        mode_str = "測試模式 (TEST)" if is_test_mode else "訓練模式 (TRAIN)"
        report_lines = [
            "="*60,
            f"📊 混合控制詳細分析報告 - {mode_str}",
            f"📝 原始日誌: {log_file}",
            "="*60,
            "\n🎨 【顏色代表意義說明】:",
            "  ● 紫色區域 (Purple)：GA 接管 - 偵測到下游交通癱瘓 (Downstream Jam)",
            "  ● 灰色區域 (Gray)  ：GA 接管 - RL 連續 20 次決策錯誤 (Penalty Drops)",
            "\n" + "-"*60
        ]
        
        # 下游癱瘓區
        if ga_jam_periods:
            total_jam = sum(e - s for s, e in ga_jam_periods)
            report_lines.append(f"🚨 下游癱瘓 (紫色區域) 觸發: {len(ga_jam_periods)} 次")
            report_lines.append(f"⏱️ 觸發時間點: {', '.join([str(p[0])+'s' for p in ga_jam_periods])}")
            report_lines.append(f"⏳ 總計疏導時長: {total_jam} 秒 (佔 {(total_jam/current_step)*100:.2f}%)")
        else:
            report_lines.append("✅ 完美！本次運行無下游癱瘓。")
            
        report_lines.append("-" * 60)
            
        # 連續掉分區
        if ga_penalty_periods:
            total_pen = sum(e - s for s, e in ga_penalty_periods)
            report_lines.append(f"📉 連續掉分 (灰色區域) 觸發: {len(ga_penalty_periods)} 次")
            report_lines.append(f"⏱️ 觸發時間點: {', '.join([str(p[0])+'s' for p in ga_penalty_periods])}")
            report_lines.append(f"⏳ 總計接管時長: {total_pen} 秒")
        else:
            report_lines.append("✅ 完美！本次運行無連續掉分觸發。")
        report_lines.append("="*60)

        full_report = "\n".join(report_lines)
        print("\n" + full_report + "\n")

        # 輸出到 TXT
        report_filename = f"report_{log_file.replace('.txt', '')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"📄 詳細報告已寫入: '{report_filename}'")

        # ==========================================
        # 繪圖部分
        # ==========================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        main_title = "Testing Performance" if is_test_mode else "Training Trend"
        ax1.set_title(f'RL {main_title} - Reward Analysis ({log_file})')
        
        ax1.plot(steps, rewards, label='Reward per 5s Step', color='deepskyblue', alpha=0.3) 
        if len(rewards) >= 20:
            mv = np.convolve(rewards, np.ones(20), 'valid') / 20
            ax1.plot(steps[19:], mv, label='20-Step Moving Average', color='red', linewidth=2)
        
        ax2.plot(steps, drops, label='Penalty Count (Drops)', color='orange')
        ax2.axhline(y=20, color='red', linestyle='--', label='GA Threshold (20)')
        
        # 繪製背景色塊與圖例
        jam_label, pen_label = False, False
        for (s, e) in ga_jam_periods:
            ax1.axvspan(s, e, color='purple', alpha=0.15, label='GA (Downstream Jam)' if not jam_label else "")
            ax2.axvspan(s, e, color='purple', alpha=0.15)
            jam_label = True
        for (s, e) in ga_penalty_periods:
            ax1.axvspan(s, e, color='gray', alpha=0.25, label='GA (20 Penalty Drops)' if not pen_label else "")
            ax2.axvspan(s, e, color='gray', alpha=0.25)
            pen_label = True

        ax1.legend(loc='upper right')
        ax1.grid(True)
        ax2.grid(True)
        plt.tight_layout()
        
        prefix = "test" if is_test_mode else "train"
        plt.savefig(f'{prefix}_result_{log_file.replace(".txt", "")}.png')
        print(f"📈 圖表已儲存為 PNG 檔。  {prefix}_result_{log_file.replace(".txt", "")}.png")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == '__main__':
    # 請將此處換成你想分析的完整 log 檔名
    plot_log_data("execute_RL_20260225_0959.txt")