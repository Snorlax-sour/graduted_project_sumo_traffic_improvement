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
    collision_count = 0  # 👑 新增車禍計數
    is_test_mode = False 
    
    pending_jam, pending_penalty, pending_end = False, False, False
    jam_start, penalty_start = None, None
    current_step = 0
    end_time = None 
    has_rl_penalty_text = False 
    
    pattern = re.compile(r"時間:\s*(\d+)s \| 綠燈.*? \| \d+秒獎勵:\s*(-?\d+\.\d+) \| 掉分:\s*(\d+)/20 \| Epsilon:\s*(\d+\.\d+)")
    end_pattern = re.compile(r"於第\s*(\d+)\s*秒提早結束") 

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 自動判定模式
                if "[TEST]" in line or "BASELINE" in log_file.upper() or "GATEST" in log_file.upper():
                    is_test_mode = True

                if "💥 [REAL_COLLISION]" in line:
                    collision_count += 1  # 👑 抓到車禍標籤

                end_match = end_pattern.search(line)
                if end_match:
                    end_time = int(end_match.group(1))

                if "下游癱瘓" in line:
                    pending_jam = True
                elif "觸發 GA 保護機制" in line: 
                    pending_penalty = True
                    has_rl_penalty_text = True 
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
                        if jam_start is not None: ga_jam_periods.append((jam_start, current_step))
                        if penalty_start is not None: ga_penalty_periods.append((penalty_start, current_step))
                        pending_end = False
                        jam_start, penalty_start = None, None

                    if pending_jam:
                        jam_start = current_step
                        pending_jam = False
                    if pending_penalty:
                        penalty_start = current_step
                        pending_penalty = False
                        
        if jam_start: ga_jam_periods.append((jam_start, current_step))
        if penalty_start: ga_penalty_periods.append((penalty_start, current_step))

        # Baseline 崩潰區間自動推算 (如果沒有 RL 標籤)
        if not has_rl_penalty_text and drops and max(drops) >= 20:
            ga_penalty_periods = [] 
            start_s = None
            for s, d in zip(steps, drops):
                if d >= 20 and start_s is None: start_s = s
                elif d < 20 and start_s is not None:
                    ga_penalty_periods.append((start_s, s))
                    start_s = None
            if start_s: ga_penalty_periods.append((start_s, steps[-1]))

        if not steps:
            print(f"❌ 略過: [{log_file}] 數據不足。")
            return

        # =================報告生成=================
        mode_str = "測試模式 (TEST)" if is_test_mode else "訓練模式 (TRAIN)"
        total_jam = sum(e - s for s, e in ga_jam_periods)
        total_pen = sum(e - s for s, e in ga_penalty_periods)
        
        report_lines = [
            "="*60, f"📊 交通控制分析報告 - {mode_str}", f"📝 日誌: {log_file}",
            f"🏁 模擬耗時: {current_step} 秒" + (f" (提早結束於 {end_time}s)" if end_time else ""),
            "="*60,
            f"🚨 下游癱瘓: {len(ga_jam_periods)} 次 | 總時長: {total_jam}s ({(total_jam/current_step)*100:.2f}%)",
            f"📉 連續掉分: {len(ga_penalty_periods)} 次 | 總時長: {total_pen}s ({(total_pen/current_step)*100:.2f}%)",
            f"💥 真實車禍: {collision_count} 次", # 👑 顯示在文字報告
            "="*60
        ]
        full_report = "\n".join(report_lines)
        print(full_report)
        with open(f"report_{log_file.replace('.txt', '')}.txt", 'w', encoding='utf-8') as f: f.write(full_report)

        # =================繪圖=================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax1.set_title(f'Traffic Performance Analysis ({log_file})')
        ax1.plot(steps, rewards, color='deepskyblue', alpha=0.3, label='Step Reward')
        if len(rewards) >= 20:
            mv = np.convolve(rewards, np.ones(20), 'valid') / 20
            ax1.plot(steps[19:], mv, color='red', linewidth=2, label='Moving Avg')
            
        ax1_twin = ax1.twinx()
        ax1_twin.plot(steps, epsilons, color='blue', linestyle=':', label='Epsilon')
        ax1_twin.set_ylabel(r'Epsilon ($\epsilon$)')
        
        ax2.plot(steps, drops, color='orange', label='Penalty Count')
        ax2.axhline(y=20, color='red', linestyle='--')
        
        # 背景色塊
        for (s, e) in ga_jam_periods:
            ax1.axvspan(s, e, color='purple', alpha=0.15)
            ax2.axvspan(s, e, color='purple', alpha=0.15)
        for (s, e) in ga_penalty_periods:
            ax1.axvspan(s, e, color='gray', alpha=0.25)
            ax2.axvspan(s, e, color='gray', alpha=0.25)

        # 👑 右側統計面板加強版
        jam_pct = (total_jam/current_step)*100
        pen_pct = (total_pen/current_step)*100
        stats_text = (
            f"--- Stats ---\n"
            f"Steps: {current_step}s\n"
            f"Collisions: {collision_count}\n" # 👑 顯示在圖表
            f"Jam: {jam_pct:.1f}%\n"
            f"Penalty: {pen_pct:.1f}%"
        )
        ax1.text(1.04, 0.5, stats_text, transform=ax1.transAxes, bbox=dict(facecolor='whitesmoke', alpha=0.9))

        ax1.legend(loc='upper left', bbox_to_anchor=(1.04, 1)); ax2.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(f'result_{log_file.replace(".txt", "")}.png', bbox_inches='tight')
        plt.close()

    except Exception as e: print(f"❌ 錯誤: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('log_files', nargs='+'); args = parser.parse_args()
    for f in args.log_files:
        for file in glob.glob(f):
            if not file.startswith("report_"): plot_log_data(file)