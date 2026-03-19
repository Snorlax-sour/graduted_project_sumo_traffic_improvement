import os
import re
import argparse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
from matplotlib.lines import Line2D 
import numpy as np 
import glob

def get_zone_type(step, jam_periods, penalty_periods):
    """判定特定時間點屬於哪種區間"""
    for i, (s, e) in enumerate(jam_periods, 1):
        if s <= step <= e: return f"紫色區塊 {i} (下游癱瘓)"
    for i, (s, e) in enumerate(penalty_periods, 1):
        if s <= step <= e: return f"灰色區塊 {i} (嚴重失控)"
    return "正常區間"

def process_episode_data(lines, log_file, ep_label=""):
    """獨立處理「單一局數 (Episode)」的數據並產出圖表與報告"""
    steps, rewards, drops, epsilons = [], [], [], [] 
    jam_periods, penalty_periods = [], []
    collision_times, rescue_times, removal_times = [], [], []
    
    is_test_mode = False 
    is_rl_mode = "RL" in os.path.basename(log_file).upper()  
    
    pending_jam, pending_penalty = False, False
    jam_start, penalty_start = None, None
    current_step = 0
    end_time = None 
    final_reward = None  
    has_rl_penalty_text = False
    
    # --- 正規表示法 ---
    pattern = re.compile(r"時間:\s*(\d+)s \| .*?10秒獎勵:\s*(-?\d+\.\d+) \| 掉分:\s*(\d+)/20 \| Epsilon:\s*(\d+\.\d+)")
    collision_pattern = re.compile(r"💥 \[REAL_COLLISION\]|Collision")
    rescue_pattern = re.compile(r"\[路口死鎖救援\]|瞬移到|向前推進")
    removal_pattern = re.compile(r"\[路口死鎖移除\]|強制移除|瞬移失敗")
    reward_pattern = re.compile(r"最終累積獎勵:\s*(-?\d+(?:\.\d+)?)")
    end_pattern = re.compile(r"於第\s*(\d+)\s*秒提早結束") 

    for line in lines:
        if "[TEST]" in line or "BASELINE" in log_file.upper() or "GATEST" in log_file.upper():
            is_test_mode = True

        # 偵測事件
        if removal_pattern.search(line): removal_times.append(current_step)
        elif rescue_pattern.search(line): rescue_times.append(current_step)
        if collision_pattern.search(line): collision_times.append(current_step)

        match = pattern.search(line)
        if match:
            current_step = int(match.group(1))
            steps.append(current_step); rewards.append(float(match.group(2)))
            d_val = int(match.group(3)); drops.append(d_val); epsilons.append(float(match.group(4)))
            
            if "下游癱瘓" in line:
                if not pending_jam: jam_start = current_step; pending_jam = True
            elif pending_jam: jam_periods.append((jam_start, current_step)); pending_jam = False
                
            if "嚴重失控" in line or d_val >= 20:
                if not penalty_start: penalty_start = current_step; has_rl_penalty_text = True
            elif penalty_start and d_val < 20:
                penalty_periods.append((penalty_start, current_step)); penalty_start = None

        reward_match = reward_pattern.search(line)
        if reward_match: final_reward = float(reward_match.group(1))
        end_match = end_pattern.search(line)
        if end_match: end_time = int(end_match.group(1))

    if jam_start: jam_periods.append((jam_start, current_step))
    if penalty_start: penalty_periods.append((penalty_start, current_step))

    if not steps: return False

    # --- 計算統計與佔比 ---
    tot_col = len(collision_times); tot_rem = len(removal_times); tot_res = len(rescue_times)
    total_jam_time = sum(e - s for s, e in jam_periods)
    total_pen_time = sum(e - s for s, e in penalty_periods)
    
    jam_pct = (total_jam_time / max(1, current_step)) * 100
    pen_pct = (total_pen_time / max(1, current_step)) * 100

    mode_str = "測試模式 (TEST)" if is_test_mode else "訓練模式 (TRAIN)"
    pure_name = os.path.basename(log_file).replace('.txt', '')
    
    # 產出區段明細字串
    def get_interval_details(periods, collisions, removals, rescues, total_c, total_rem, total_res):
        details = []
        for i, (s, e) in enumerate(periods, 1):
            c_cnt = sum(1 for c in collisions if s <= c <= e); rem_cnt = sum(1 for r in removals if s <= r <= e); res_cnt = sum(1 for res in rescues if s <= res <= e)
            c_pct = (c_cnt / total_c * 100) if total_c > 0 else 0.0
            rem_pct = (rem_cnt / total_rem * 100) if total_rem > 0 else 0.0
            res_pct = (res_cnt / total_res * 100) if total_res > 0 else 0.0
            details.append(f"  🔸 第 {s}s ~ {e}s (持續 {e-s}s，區間內車禍: {c_cnt} 次 ({c_pct:.1f}%), 區間內移除: {rem_cnt}次 ({rem_pct:.1f}%), 區間內移動: {res_cnt}次 ({res_pct:.1f}%))")
        return "\n".join(details) if details else "  無"

    # --- 組合完整報告 ---
    report_content = [
        "="*60,
        f"📊 交通控制分析報告 - {mode_str} {ep_label.replace('_', '')}".strip(),
        f"📝 日誌: {os.path.basename(log_file)}",
        f"🏁 模擬耗時: {current_step}s" + (f" (提早結束於 {end_time}s)" if end_time else ""),
        f"💰 最終累積獎勵: {final_reward if final_reward is not None else sum(rewards):.2f}",
        f"移除車輛共 {tot_rem} 次",
        f"移動車輛共 {tot_res} 次",
        "="*60,
        f"🚨 下游癱瘓 (紫色): {len(jam_periods)} 次 | 總時長: {total_jam_time}s ({jam_pct:.2f}%)",
        "   [具體發生時間]:", get_interval_details(jam_periods, collision_times, removal_times, rescue_times, tot_col, tot_rem, tot_res),
        "-"*60,
        f"📉 嚴重失控 (灰色): {len(penalty_periods)} 次 | 總時長: {total_pen_time}s ({pen_pct:.2f}%)",
        "   [具體發生時間]:", get_interval_details(penalty_periods, collision_times, removal_times, rescue_times, tot_col, tot_rem, tot_res),
        "-"*60,
        f"💥 真實車禍紀錄 (共 {tot_col} 次):",
        "\n".join([f"  🔸 車禍時間: {t}s ({get_zone_type(t, jam_periods, penalty_periods)})" for t in collision_times]) if collision_times else "  無",
        "-"*60,
        f"🚑 移動車輛 (瞬移救援) 明細 (共 {tot_res} 次):",
        "\n".join([f"  🔸 救援時間: {t}s ({get_zone_type(t, jam_periods, penalty_periods)})" for t in rescue_times]) if rescue_times else "  無",
        "-"*60,
        f"💀 移除車輛 (強制移除) 明細 (共 {tot_rem} 次):",
        "\n".join([f"  🔸 移除時間: {t}s ({get_zone_type(t, jam_periods, penalty_periods)})" for t in removal_times]) if removal_times else "  無",
        "="*60
    ]
    
    # 存檔報告 (附加 ep_label，如果是單局則無後綴)
    with open(f"report_{pure_name}{ep_label}.txt", 'w', encoding='utf-8') as f: 
        f.write("\n".join(report_content))
        print(f"\n✅ 成功產出文字報告: report_{pure_name}{ep_label}.txt")

    # ================= 繪圖 =================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    title_suffix = f" ({ep_label.replace('_', '')})" if ep_label else ""
    ax1.set_title(f'Performance Analysis: {pure_name}{title_suffix}', fontsize=14, fontweight='bold')
    
    ax1.plot(steps, rewards, color='deepskyblue', alpha=0.3, label='10s Reward')
    if len(rewards) >= 20:
        mv = np.convolve(rewards, np.ones(20), 'valid') / 20
        ax1.plot(steps[19:], mv, color='red', linewidth=2, label='Reward Trend (Moving Avg)')

    ax1.scatter(collision_times, [min(rewards)]*len(collision_times), color='red', marker='x', s=40, label='Collision (x)')
    ax1.scatter(rescue_times, [min(rewards)*1.05]*len(rescue_times), color='green', marker='o', s=30, label='Rescue Move (o)')
    ax1.scatter(removal_times, [min(rewards)*1.15]*len(removal_times), color='black', marker='^', s=60, label='Vehicle Removal (^)')

    ax2.plot(steps, drops, color='orange', label='Waiting Vehicle Count')
    ax2.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Severe Jam Threshold (20)') 

    for (s, e) in jam_periods:
        ax1.axvspan(s, e, color='purple', alpha=0.15)
        ax2.axvspan(s, e, color='purple', alpha=0.15)
    for (s, e) in penalty_periods:
        ax1.axvspan(s, e, color='gray', alpha=0.25)
        ax2.axvspan(s, e, color='gray', alpha=0.25)

    jam_patch = mpatches.Patch(color='purple', alpha=0.15, label='Downstream Jam (Purple Zone)')
    fail_patch = mpatches.Patch(color='gray', alpha=0.25, label='Severe Failure (Gray Zone)')
    col_mark = Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8, label='Collision (x)')
    res_mark = Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=8, label='Rescue Move (o)')
    rem_mark = Line2D([0], [0], color='black', marker='^', linestyle='None', markersize=8, label='Vehicle Removal (^)')
    
    ax1.legend(handles=[jam_patch, fail_patch, col_mark, res_mark, rem_mark], loc='upper left', bbox_to_anchor=(1.02, 1), title="Legend & Symbols")
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Metrics")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"result_{pure_name}{ep_label}.png", dpi=150)
    plt.close()
    print(f"✅ 成功產出圖表: result_{pure_name}{ep_label}.png")
    return True

def plot_log_data(log_file):
    """讀取檔案並切分 Episode"""
    if not os.path.exists(log_file):
        print(f"⚠️ 找不到檔案: {log_file}")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    episodes = []
    current_ep_lines = []
    current_ep_label = ""
    
    for line in lines:
        if "🏁 正在啟動第" in line:
            # 遇到新的 Episode，把之前的數據封存
            if current_ep_lines:
                episodes.append((current_ep_label, current_ep_lines))
            current_ep_lines = [line]
            
            # 抓取局數編號
            ep_match = re.search(r"🏁 正在啟動第\s*(\d+)\s*局模擬", line)
            if ep_match:
                current_ep_label = f"_Ep{ep_match.group(1)}"
            else:
                current_ep_label = "_EpX"
        else:
            current_ep_lines.append(line)
            
    # 把最後一個 Episode 封存
    if current_ep_lines:
        episodes.append((current_ep_label, current_ep_lines))
        
    # 開始逐局處理
    processed_count = 0
    for ep_label, ep_lines in episodes:
        if process_episode_data(ep_lines, log_file, ep_label):
            processed_count += 1
            
    if processed_count == 0:
        print(f"⚠️ 檔案 {log_file} 中找不到有效的模擬數據。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('log_files', nargs='+'); args = parser.parse_args()
    for f in args.log_files:
        for file in glob.glob(f): 
            if not os.path.basename(file).startswith(("report_", "result_")): 
                plot_log_data(file)