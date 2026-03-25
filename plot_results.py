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
    """獨立處理單一局數，並解析四維度懲罰"""
    steps, rewards, epsilons = [], [], []
    p_wait_list, p_junc_list, p_down_list, p_col_list = [], [], [], []
    
    jam_periods, penalty_periods = [], []
    collision_times, rescue_times, removal_times = [], [], []
    
    is_test_mode = "TEST" in log_file.upper() or "BASELINE" in log_file.upper()
    
    pending_jam, pending_penalty = False, False
    jam_start, penalty_start = None, None
    current_step = 0
    end_time = None 
    final_reward = None  
    
    # 👑 新版正規表示法：抓取獨立的四個懲罰項
    # 注意：使用 .*? 跳過 RL 模式獨有的「切換罰」欄位，讓 BASELINE/GA_TEST/RL 的 Log 都能通用
    pattern = re.compile(r"時間:\s*(\d+)s \| .*?10秒獎勵:\s*(-?\d+\.\d+) \| 延遲罰:\s*(\d+\.\d+).*?路口罰:\s*(\d+\.\d+) \| 下游罰:\s*(\d+\.\d+) \| 車禍罰:\s*(\d+\.\d+) \| Epsilon:\s*(\d+\.\d+)")
    
    collision_pattern = re.compile(r"💥 \[REAL_COLLISION\]|Collision")
    rescue_pattern = re.compile(r"\[路口死鎖救援\]|瞬移到|向前推進")
    removal_pattern = re.compile(r"\[路口死鎖移除\]|強制移除|瞬移失敗")
    reward_pattern = re.compile(r"最終累積獎勵:\s*(-?\d+(?:\.\d+)?)")
    end_pattern = re.compile(r"於第\s*(\d+)\s*秒提早結束") 

    for line in lines:
        if collision_pattern.search(line): collision_times.append(current_step)
        if rescue_pattern.search(line): rescue_times.append(current_step)
        if removal_pattern.search(line): removal_times.append(current_step)

        match = pattern.search(line)
        if match:
            current_step = int(match.group(1))
            steps.append(current_step)
            rewards.append(float(match.group(2)))
            
            p_w = float(match.group(3))
            p_j = float(match.group(4))
            p_d = float(match.group(5))
            p_c = float(match.group(6))
            
            p_wait_list.append(p_w)
            p_junc_list.append(p_j)
            p_down_list.append(p_d)
            p_col_list.append(p_c)
            epsilons.append(float(match.group(7)))
            
            # 👑 基於真實物理懲罰自動標記「崩潰區間」
            if p_d > 0:
                if not pending_jam: jam_start = current_step; pending_jam = True
            elif pending_jam: jam_periods.append((jam_start, current_step)); pending_jam = False
                
            if p_j > 0 or p_c > 0: # 只要路口卡死或車禍，就是嚴重失控(灰色)
                if not penalty_start: penalty_start = current_step; pending_penalty = True
            elif pending_penalty:
                penalty_periods.append((penalty_start, current_step)); pending_penalty = False; penalty_start = None

        reward_match = reward_pattern.search(line)
        if reward_match: final_reward = float(reward_match.group(1))
        end_match = end_pattern.search(line)
        if end_match: end_time = int(end_match.group(1))

    if pending_jam: jam_periods.append((jam_start, current_step))
    if pending_penalty: penalty_periods.append((penalty_start, current_step))

    if not steps: return False

    tot_col = len(collision_times); tot_rem = len(removal_times); tot_res = len(rescue_times)
    total_jam_time = sum(e - s for s, e in jam_periods)
    total_pen_time = sum(e - s for s, e in penalty_periods)
    
    jam_pct = (total_jam_time / max(1, current_step)) * 100
    pen_pct = (total_pen_time / max(1, current_step)) * 100

    mode_str = "測試模式 (TEST)" if is_test_mode else "訓練模式 (TRAIN)"
    pure_name = os.path.basename(log_file).replace('.txt', '')
    
    def get_interval_details(periods, collisions, removals, rescues, total_c, total_rem, total_res):
        details = []
        for i, (s, e) in enumerate(periods, 1):
            c_cnt = sum(1 for c in collisions if s <= c <= e); rem_cnt = sum(1 for r in removals if s <= r <= e); res_cnt = sum(1 for res in rescues if s <= res <= e)
            c_pct = (c_cnt / total_c * 100) if total_c > 0 else 0.0
            rem_pct = (rem_cnt / total_rem * 100) if total_rem > 0 else 0.0
            res_pct = (res_cnt / total_res * 100) if total_res > 0 else 0.0
            details.append(f"  🔸 第 {s}s ~ {e}s (持續 {e-s}s，區間內車禍: {c_cnt} 次 ({c_pct:.1f}%), 區間內移除: {rem_cnt}次 ({rem_pct:.1f}%), 區間內移動: {res_cnt}次 ({res_pct:.1f}%))")
        return "\n".join(details) if details else "  無"

    # ================= 嚴格遵守原始文字報告格式 =================
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
    
    with open(f"report_{pure_name}{ep_label}.txt", 'w', encoding='utf-8') as f: 
        f.write("\n".join(report_content))
        print(f"\n✅ 成功產出文字報告: report_{pure_name}{ep_label}.txt")

    # ================= 繪圖 (嚴格保持原有風格) =================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    title_suffix = f" ({ep_label.replace('_', '')})" if ep_label else ""
    ax1.set_title(f'Performance Analysis: {pure_name}{title_suffix}', fontsize=14, fontweight='bold')
    
    # [原封不動] 上半部圖表：總分、移動平均、標記點
    ax1.plot(steps, rewards, color='deepskyblue', alpha=0.3, label='10s Reward')
    if len(rewards) >= 20:
        mv = np.convolve(rewards, np.ones(20), 'valid') / 20
        ax1.plot(steps[19:], mv, color='red', linewidth=2, label='Reward Trend (Moving Avg)')

    ax1.scatter(collision_times, [min(rewards)]*len(collision_times), color='red', marker='x', s=40, label='Collision (x)')
    ax1.scatter(rescue_times, [min(rewards)*1.05]*len(rescue_times), color='green', marker='o', s=30, label='Rescue Move (o)')
    ax1.scatter(removal_times, [min(rewards)*1.15]*len(removal_times), color='black', marker='^', s=60, label='Vehicle Removal (^)')

    for (s, e) in jam_periods: ax1.axvspan(s, e, color='purple', alpha=0.15)
    for (s, e) in penalty_periods: ax1.axvspan(s, e, color='gray', alpha=0.25)

    jam_patch = mpatches.Patch(color='purple', alpha=0.15, label='Downstream Jam (Purple Zone)')
    fail_patch = mpatches.Patch(color='gray', alpha=0.25, label='Severe Failure (Gray Zone)')
    col_mark = Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8, label='Collision (x)')
    res_mark = Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=8, label='Rescue Move (o)')
    rem_mark = Line2D([0], [0], color='black', marker='^', linestyle='None', markersize=8, label='Vehicle Removal (^)')
    
    ax1.legend(handles=[jam_patch, fail_patch, col_mark, res_mark, rem_mark], loc='upper left', bbox_to_anchor=(1.02, 1), title="Legend & Symbols")

    # 👑 【核心重構】下半部圖表：從單線改成四維度堆疊面積圖
    ax2.stackplot(steps, p_wait_list, p_down_list, p_junc_list, p_col_list,
                  labels=['Delay & Queue', 'Downstream Jam (Purple)', 'Junction Blocking (Gray)', 'Collision (Red)'],
                  colors=['#FFCC99', '#DDA0DD', '#A9A9A9', '#FF9999'], alpha=0.8)

    # 保留下半部的背景色對照
    for (s, e) in jam_periods: ax2.axvspan(s, e, color='purple', alpha=0.15)
    for (s, e) in penalty_periods: ax2.axvspan(s, e, color='gray', alpha=0.25)

    ax2.set_ylabel("Penalty Breakdown (Scores)")
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Penalty Components")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"result_{pure_name}{ep_label}.png", dpi=150)
    plt.close()
    print(f"✅ 成功產出圖表: result_{pure_name}{ep_label}.png")
    return True

def plot_log_data(log_file):
    if not os.path.exists(log_file): return
    with open(log_file, 'r', encoding='utf-8') as f: lines = f.readlines()
        
    episodes = []
    current_ep_lines, current_ep_label = [], ""
    for line in lines:
        if "🏁 正在啟動第" in line:
            if current_ep_lines: episodes.append((current_ep_label, current_ep_lines))
            current_ep_lines = [line]
            ep_match = re.search(r"🏁 正在啟動第\s*(\d+)\s*局模擬", line)
            current_ep_label = f"_Ep{ep_match.group(1)}" if ep_match else "_EpX"
        else:
            current_ep_lines.append(line)
            
    if current_ep_lines: episodes.append((current_ep_label, current_ep_lines))
        
    for ep_label, ep_lines in episodes:
        process_episode_data(ep_lines, log_file, ep_label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('log_files', nargs='+'); args = parser.parse_args()
    for f in args.log_files:
        for file in glob.glob(f): 
            if not os.path.basename(file).startswith(("report_", "result_")): 
                plot_log_data(file)