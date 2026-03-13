import re
import argparse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import numpy as np 
import glob

def plot_log_data(log_file):
    steps, rewards, drops, epsilons = [], [], [], [] 
    ga_jam_periods, ga_penalty_periods = [], []
    collision_count = 0  
    collision_times = []  
    is_test_mode = False 
    
    pending_jam, pending_penalty, pending_end = False, False, False
    jam_start, penalty_start = None, None
    current_step = 0
    end_time = None 
    has_rl_penalty_text = False 
    final_reward = None  # 👑 新增：儲存最終累積獎勵
    
    pattern = re.compile(r"時間:\s*(\d+)s \| 綠燈.*? \| \d+秒獎勵:\s*(-?\d+\.\d+) \| 掉分:\s*(\d+)/20 \| Epsilon:\s*(\d+\.\d+)")
    end_pattern = re.compile(r"於第\s*(\d+)\s*秒提早結束") 
    # 👑 新增正則表達式來抓取最終結算的獎勵
    reward_pattern = re.compile(r"最終累積獎勵:\s*(-?\d+(?:\.\d+)?)")

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "[TEST]" in line or "BASELINE" in log_file.upper() or "GATEST" in log_file.upper():
                    is_test_mode = True

                # 抓取最終累積獎勵
                reward_match = reward_pattern.search(line)
                if reward_match:
                    final_reward = float(reward_match.group(1))

                # 抓到車禍標籤並提取發生時間
                if "💥 [REAL_COLLISION]" in line:
                    collision_count += 1
                    col_match = re.search(r"Step:\s*(\d+)", line)
                    if col_match:
                        collision_times.append(int(col_match.group(1)))

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

        # Baseline 崩潰區間自動推算 
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

        # 👑 如果 Log 檔案裡面沒有印出「最終累積獎勵」，我們就自動幫它把所有收集到的 rewards 加總
        if final_reward is None and rewards:
            final_reward = sum(rewards)

        # ================= 👑 車禍時間標註與【數量統計】分析 =================
        collision_details = []
        jam_col_cnt = 0  
        pen_col_cnt = 0  

        for c_time in collision_times:
            tags = []
            in_jam, in_pen = False, False
            
            # 比對紫色(下游癱瘓)
            for i, (s, e) in enumerate(ga_jam_periods, 1):
                if s <= c_time <= e:
                    tags.append(f"紫色區塊 {i}")
                    in_jam = True
                    break
                    
            # 比對灰色(連續掉分)
            for i, (s, e) in enumerate(ga_penalty_periods, 1):
                if s <= c_time <= e:
                    tags.append(f"灰色區塊 {i}")
                    in_pen = True
                    break
            
            # 統計累加
            if in_jam: jam_col_cnt += 1
            if in_pen: pen_col_cnt += 1

            if tags:
                collision_details.append(f"  🔸 車禍時間: {c_time}s ({' & '.join(tags)} 過程中發生)")
            else:
                collision_details.append(f"  🔸 車禍時間: {c_time}s (正常區間)")

        collision_report_str = "\n".join(collision_details) if collision_details else "  無"

        # =================報告生成=================
        mode_str = "測試模式 (TEST)" if is_test_mode else "訓練模式 (TRAIN)"
        total_jam = sum(e - s for s, e in ga_jam_periods)
        total_pen = sum(e - s for s, e in ga_penalty_periods)
        
        # 計算每個紫色區塊內的車禍數
        jam_details_list = []
        for s, e in ga_jam_periods:
            cnt = sum(1 for c in collision_times if s <= c <= e)
            jam_details_list.append(f"  🔸 第 {s}s ~ {e}s (持續 {e-s}s，區段內車禍發生次數：{cnt} 次)")
        jam_details = "\n".join(jam_details_list) if jam_details_list else "  無"

        # 計算每個灰色區塊內的車禍數
        pen_details_list = []
        for s, e in ga_penalty_periods:
            cnt = sum(1 for c in collision_times if s <= c <= e)
            pen_details_list.append(f"  🔸 第 {s}s ~ {e}s (持續 {e-s}s，區段內車禍發生次數：{cnt} 次)")
        pen_details = "\n".join(pen_details_list) if pen_details_list else "  無"

        final_reward_str = f"{final_reward:.2f}" if final_reward is not None else "未知"

        report_lines = [
            "="*60, f"📊 交通控制分析報告 - {mode_str}", f"📝 日誌: {log_file}",
            f"🏁 模擬耗時: {current_step} 秒" + (f" (提早結束於 {end_time}s)" if end_time else ""),
            f"💰 最終累積獎勵: {final_reward_str}",  # 👑 顯示在報告頭部
            "="*60,
            f"🚨 下游癱瘓: {len(ga_jam_periods)} 次 | 總時長: {total_jam}s ({(total_jam/max(1, current_step))*100:.2f}%)",
            "   [具體發生時間]:",
            jam_details,
            "-"*60,
            f"📉 連續掉分: {len(ga_penalty_periods)} 次 | 總時長: {total_pen}s ({(total_pen/max(1, current_step))*100:.2f}%)",
            "   [具體發生時間]:",
            pen_details,
            "-"*60,
            f"💥 真實車禍: 共 {collision_count} 次 (灰色掉分區: {pen_col_cnt} 次 | 紫色癱瘓區: {jam_col_cnt} 次)", 
            "   [具體車禍時間與狀態]:",
            collision_report_str,
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
        
        for (s, e) in ga_jam_periods:
            ax1.axvspan(s, e, color='purple', alpha=0.15)
            ax2.axvspan(s, e, color='purple', alpha=0.15)
        for (s, e) in ga_penalty_periods:
            ax1.axvspan(s, e, color='gray', alpha=0.25)
            ax2.axvspan(s, e, color='gray', alpha=0.25)

        jam_patch = mpatches.Patch(color='purple', alpha=0.15, label='Downstream Jam (Purple)')
        pen_patch = mpatches.Patch(color='gray', alpha=0.25, label='Penalty Period (Gray)')

        jam_pct = (total_jam/max(1, current_step))*100
        pen_pct = (total_pen/max(1, current_step))*100
        
        # 👑 將最終獎勵加入圖片的 Stats 面板中
        stats_text = (
            f"--- Stats ---\n"
            f"Steps: {current_step}s\n"
            f"Reward: {final_reward_str}\n" 
            f"Total Colls: {collision_count}\n" 
            f"Colls(Gray): {pen_col_cnt}\n"
            f"Colls(Purple): {jam_col_cnt}\n"
            f"Jam: {jam_pct:.1f}%\n"
            f"Penalty: {pen_pct:.1f}%"
        )
        ax1.text(1.04, 0.5, stats_text, transform=ax1.transAxes, bbox=dict(facecolor='whitesmoke', alpha=0.9))

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles_twin, labels_twin = ax1_twin.get_legend_handles_labels()
        ax1.legend(handles1 + handles_twin + [jam_patch, pen_patch], 
                   labels1 + labels_twin + ['Downstream Jam (Purple)', 'Penalty Period (Gray)'], 
                   loc='upper left', bbox_to_anchor=(1.04, 1))

        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles2 + [jam_patch, pen_patch], 
                   labels2 + ['Downstream Jam (Purple)', 'Penalty Period (Gray)'], 
                   loc='upper left', bbox_to_anchor=(1.04, 1))

        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(f'result_{log_file.replace(".txt", "")}.png', bbox_inches='tight')
        plt.close()

    except Exception as e: print(f"❌ 錯誤: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('log_files', nargs='+'); args = parser.parse_args()
    for f in args.log_files:
        for file in glob.glob(f):
            if not file.startswith("report_"): plot_log_data(file)