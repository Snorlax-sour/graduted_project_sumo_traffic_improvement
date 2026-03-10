import re
import argparse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np 
import glob

def plot_log_data(log_file):
    steps, rewards, drops, epsilons = [], [], [], [] 
    ga_jam_periods, ga_penalty_periods = [], []
    is_test_mode = False 
    
    pending_jam, pending_penalty, pending_end = False, False, False
    jam_start, penalty_start = None, None
    current_step = 0
    end_time = None 
    
    # 👑 新增：追蹤是否有印出 RL 專屬的觸發字眼
    has_rl_penalty_text = False 
    
    pattern = re.compile(r"時間:\s*(\d+)s \| 綠燈.*? \| \d+秒獎勵:\s*(-?\d+\.\d+) \| 掉分:\s*(\d+)/20 \| Epsilon:\s*(\d+\.\d+)")
    end_pattern = re.compile(r"於第\s*(\d+)\s*秒提早結束") 

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "[TEST]" in line:
                    is_test_mode = True

                end_match = end_pattern.search(line)
                if end_match:
                    end_time = int(end_match.group(1))

                if "下游癱瘓" in line:
                    pending_jam = True
                elif "觸發 GA 保護機制" in line: 
                    pending_penalty = True
                    has_rl_penalty_text = True  # 標記這是 RL 產生的日誌
                elif "GA 示範結束" in line:
                    pending_end = True

                match = pattern.search(line)
                if match:
                    current_step = int(match.group(1))
                    steps.append(current_step)
                    rewards.append(float(match.group(2)))
                    drops.append(int(match.group(3))) 
                    epsilons.append(float(match.group(4))) 
                    
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
                        
        if jam_start is not None: ga_jam_periods.append((jam_start, current_step))
        if penalty_start is not None: ga_penalty_periods.append((penalty_start, current_step))

        # 👑 【核心修復】：如果這是 BASELINE / GATEST (沒印觸發字眼，但掉分有超過20)
        # 我們直接用掉分的數字陣列，推算出它崩潰的區間！
        if not has_rl_penalty_text and drops and max(drops) >= 20:
            ga_penalty_periods = [] 
            start_s = None
            for s, d in zip(steps, drops):
                if d >= 20 and start_s is None:
                    start_s = s  # 開始崩潰
                elif d < 20 and start_s is not None:
                    ga_penalty_periods.append((start_s, s)) # 崩潰結束
                    start_s = None
            if start_s is not None:
                ga_penalty_periods.append((start_s, steps[-1])) # 一路崩潰到結尾

        if not steps:
            print(f"❌ 略過: [{log_file}] 找不到可解析的數據。")
            return

        # ==========================================
        # 建立報告文字內容
        # ==========================================
        mode_str = "測試模式 (TEST)" if is_test_mode else "訓練模式 (TRAIN)"
        report_lines = ["="*60, f"📊 混合控制詳細分析報告 - {mode_str}", f"📝 原始日誌: {log_file}"]
        
        if end_time:
            report_lines.append(f"🏁 模擬結束時間: 第 {end_time} 秒 (提早清空車流)")
        else:
            report_lines.append(f"🏁 模擬結束時間: 第 {current_step} 秒 (完整模擬/未提早結束)")
            
        report_lines.extend([
            "="*60,
            "\n🎨 【圖表視覺代表意義說明】:",
            "  [背景色塊 - GA 安全接管機制]",
            "  ● 紫色區域 (Purple)：偵測到下游交通癱瘓 (Downstream Jam)",
            "  ● 灰色區域 (Gray)  ：RL 連續 20 次決策錯誤 (Penalty Drops)",
            "  [線條指標 - RL 健康度監控]",
            "  ● 淺藍/紅線 (Blue/Red)：獎勵與20步移動平均線 (Reward)",
            "  ● 綠色虛線 (Green Dash)：探索率下降趨勢 (Exploration Rate ε)",
            "  ● 橘色折線 (Orange)  ：連續錯誤決策次數 (Penalty Count)",
            "\n" + "-"*60
        ])
        
        total_jam = sum(e - s for s, e in ga_jam_periods) if ga_jam_periods else 0
        total_pen = sum(e - s for s, e in ga_penalty_periods) if ga_penalty_periods else 0
        
        if ga_jam_periods:
            report_lines.append(f"🚨 下游癱瘓 (紫色區域) 觸發: {len(ga_jam_periods)} 次")
            report_lines.append(f"⏱️ 觸發時間點: {', '.join([str(p[0])+'s' for p in ga_jam_periods])}")
            report_lines.append(f"⏳ 總計疏導時長: {total_jam} 秒 (佔 {(total_jam/current_step)*100:.2f}%)")
        else:
            report_lines.append("✅ 完美！本次運行無下游癱瘓。")
            
        report_lines.append("-" * 60)
            
        if ga_penalty_periods:
            report_lines.append(f"📉 連續掉分 (灰色區域) 觸發: {len(ga_penalty_periods)} 次")
            report_lines.append(f"⏱️ 觸發時間點: {', '.join([str(p[0])+'s' for p in ga_penalty_periods])}")
            report_lines.append(f"⏳ 總計接管/崩潰時長: {total_pen} 秒 (佔 {(total_pen/current_step)*100:.2f}%)")
        else:
            report_lines.append("✅ 完美！本次運行無連續掉分觸發。")
        report_lines.append("="*60)

        full_report = "\n".join(report_lines)
        print(full_report)

        with open(f"report_{log_file.replace('.txt', '')}.txt", 'w', encoding='utf-8') as f:
            f.write(full_report)

        # ==========================================
        # 繪圖部分 
        # ==========================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        main_title = "Testing Performance" if is_test_mode else "Training Trend"
        ax1.set_title(f'RL {main_title} - Reward Analysis ({log_file})')
        
        ax1.plot(steps, rewards, label='Reward per Step', color='deepskyblue', alpha=0.3) 
        if len(rewards) >= 20:
            mv = np.convolve(rewards, np.ones(20), 'valid') / 20
            ax1.plot(steps[19:], mv, label='20-Step Moving Average', color='red', linewidth=2)
            
        ax1_twin = ax1.twinx()
        ax1_twin.plot(steps, epsilons, label='Exploration Rate (Epsilon)', color='blue', linestyle=':', linewidth=2, alpha=0.8)
        ax1_twin.set_ylabel(r'Epsilon ($\epsilon$)')
        ax1_twin.set_ylim(-0.05, 1.05)
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        
        ax2.plot(steps, drops, label='Consecutive Penalty (Drops)', color='orange')
        ax2.axhline(y=20, color='red', linestyle='--', label='GA Override Threshold (20)')
        
        for (s, e) in ga_jam_periods:
            ax1.axvspan(s, e, color='purple', alpha=0.15)
            ax2.axvspan(s, e, color='purple', alpha=0.15, label='GA (Downstream Jam)' if s == ga_jam_periods[0][0] else "")
        for (s, e) in ga_penalty_periods:
            ax1.axvspan(s, e, color='gray', alpha=0.25)
            ax2.axvspan(s, e, color='gray', alpha=0.25, label='GA (20 Penalty Drops)' if s == ga_penalty_periods[0][0] else "")

        # 圖例定位
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0)
        ax2.legend(loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0)
        
        # 統計資訊面板
        jam_pct = (total_jam / current_step) * 100 if current_step > 0 else 0
        pen_pct = (total_pen / current_step) * 100 if current_step > 0 else 0
        stats_text = (
            f"--- Simulation Stats ---\n"
            f"Total Steps: {current_step}s\n"
            f"Jam (Purple): {jam_pct:.1f}% ({total_jam}s)\n"
            f"Penalty (Gray): {pen_pct:.1f}% ({total_pen}s)"
        )
        ax1.text(1.04, 0.05, stats_text, transform=ax1.transAxes, 
                 fontsize=11, verticalalignment='bottom', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="lightgray", alpha=0.9))

        ax1.grid(True)
        ax2.grid(True)
        
        plt.tight_layout(rect=[0, 0, 0.82, 1]) 
        
        prefix = "test" if is_test_mode else "train"
        save_name = f'{prefix}_result_{log_file.replace(".txt", "")}.png'
        
        plt.savefig(save_name, bbox_inches='tight')
        plt.close(fig)
        
        print(f"📄 報告已儲存: report_{log_file.replace('.txt', '')}.txt")
        print(f"📈 圖表已儲存: {save_name}\n")

    except Exception as e:
        print(f"❌ 發生錯誤 ({log_file}): {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="📊 混合控制 (RL/GA/Baseline) 訓練日誌繪圖工具")
    parser.add_argument('log_files', nargs='+', help="要處理的日誌檔案，支援萬用字元")
    args = parser.parse_args()

    all_files = []
    for f in args.log_files:
        all_files.extend(glob.glob(f))

    all_files = list(set(all_files))

    if not all_files:
        print("❌ 找不到任何符合條件的日誌檔案！")
    else:
        for file in all_files:
            if file.startswith("report_"):
                continue
            print(f"⏳ 正在處理: {file} ...")
            plot_log_data(file)
            
        print("🎉 批次繪圖作業全部完成！")