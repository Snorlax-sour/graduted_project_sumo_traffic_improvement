import sys
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M")


# ==============================================================================
# 工具函數
# ==============================================================================

def parse_rl_filename(filepath):
    """
    從 RL 檔名解析 model_id。
    格式：execute_RL_{timestamp}_{model_id}_{mode}.txt
    範例：execute_RL_20260403_0048_20260402_train.txt
          → model_id = "20260402"
    """
    basename = os.path.basename(filepath)
    # 去掉副檔名與前綴
    name = basename.replace("execute_RL_", "").replace(".txt", "")
    # 格式：yyyymmdd_hhmm_modelid_mode
    parts = name.split("_")
    # parts[0]=yyyymmdd, parts[1]=hhmm, parts[-1]=mode, 中間都是 model_id
    if len(parts) >= 4:
        model_id = "_".join(parts[2:-1])
        return model_id
    return None





def extract_episode_penalties(content):
    """
    從檔案內容提取延遲罰數據，回傳 (penalties, mode)。
    - 訓練模式（多局）：每局延遲罰總和，mode="episode"
    - 測試模式（單局）：每個 10 秒決策點的延遲罰，mode="step"
    """
    episodes = re.split(r"🏁 正在啟動第 \d+ 局", content)[1:]
    if len(episodes) > 1:
        # 訓練模式：多局，每局加總
        penalties = []
        for ep_content in episodes:
            delays = re.findall(r"延遲罰:\s*([\d.]+)", ep_content)
            if delays:
                penalties.append(sum(float(d) for d in delays))
        return penalties, "episode"
    else:
        # 測試模式：單局，每個決策點的延遲罰
        delays = re.findall(r"延遲罰:\s*([\d.]+)", content)
        return [float(d) for d in delays], "step"


def extract_ga_data(filepath):
    """
    從 GA CSV 提取每代最佳分數，回傳 (generations, best_scores)。
    支援兩種 CSV 欄位格式：
      - 新版：generation, individual_idx, phase1, phase2, total_score, ...
      - 舊版：generation, phase1, phase2, delay, os_pid
    只接受 CSV，舊版 txt log 沒有記錄分數無法使用。
    """
    if not filepath.endswith('.csv'):
        print(f"   ⚠️ 跳過非 CSV 檔案: {os.path.basename(filepath)}")
        print(f"      GA txt log 沒有記錄分數，請改用對應的 CSV 檔案")
        return [], []

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"   ⚠️ 無法讀取 {os.path.basename(filepath)}: {e}")
        return [], []

    if df.empty:
        print(f"   ⚠️ {os.path.basename(filepath)} 是空檔案，跳過")
        return [], []

    # 自動偵測分數欄位（新版 total_score / 舊版 delay）
    if 'total_score' in df.columns:
        score_col = 'total_score'
    elif 'delay' in df.columns:
        score_col = 'delay'
    else:
        print(f"   ⚠️ {os.path.basename(filepath)} 找不到分數欄位（需要 total_score 或 delay）")
        print(f"      現有欄位: {list(df.columns)}")
        return [], []

    grouped = df.groupby('generation')[score_col].min()
    return list(grouped.index), list(grouped.values)


# ==============================================================================
# RL 模式
# ==============================================================================

def plot_rl(txt_files):
    """
    接收多個 RL log 檔案。
    - 先確認所有檔案是否來自同一個 model_id
    - 是 → 串接成單一收斂圖
    - 否 → 報錯提示，讓使用者確認
    """
    if not txt_files:
        print("❌ 沒有找到任何 RL 檔案。")
        return

    # 解析每個檔案的 model_id
    file_model_map = {}
    for f in txt_files:
        if not os.path.exists(f):
            print(f"⚠️ 找不到檔案，跳過: {f}")
            continue
        mid = parse_rl_filename(f)
        file_model_map[f] = mid

    if not file_model_map:
        print("❌ 所有檔案均不存在。")
        return

    # 確認 model_id 是否一致
    model_ids = set(file_model_map.values())
    if len(model_ids) > 1:
        print("❌ 偵測到多個不同的 model_id，無法合併：")
        for f, mid in file_model_map.items():
            print(f"   {os.path.basename(f)} → model: {mid}")
        print("請只傳入同一個 model 的訓練紀錄。")
        return

    model_id = list(model_ids)[0]
    valid_files = sorted(file_model_map.keys())  # 依檔名排序（時間序）
    print(f"✅ 確認所有檔案均來自 model: [{model_id}]，共 {len(valid_files)} 個檔案")

    # 依序串接所有局的數據
    all_penalties = []
    boundaries = []  # 記錄每個檔案結束時的 episode 編號（用於畫分界線）
    final_data_mode = "episode"  # 預設訓練模式，會被實際資料覆蓋

    for f in valid_files:
        print(f"🧠 正在讀取: {os.path.basename(f)}")
        with open(f, "r", encoding="utf-8") as fh:
            content = fh.read()
        penalties, data_mode = extract_episode_penalties(content)
        if penalties:
            all_penalties.extend(penalties)
            final_data_mode = data_mode
            boundaries.append((len(all_penalties), os.path.basename(f)))
            unit = "局" if data_mode == "episode" else "步"
            print(f"   → 讀到 {len(penalties)} {unit}，累計 {len(all_penalties)} {unit}")
        else:
            print(f"   → ⚠️ 找不到可用數據，跳過")

    if not all_penalties:
        print("❌ 所有檔案均找不到可用的 RL 訓練數據。")
        return

    # 繪圖
    plt.figure(figsize=(12, 6))
    x = range(1, len(all_penalties) + 1)
    plt.plot(x, all_penalties, alpha=0.3, color='orange', label='Episode Total Delay')

    series = pd.Series(all_penalties)
    smooth = series.rolling(window=5, min_periods=1).mean()
    plt.plot(x, smooth, color='red', linewidth=2, label='Trend (Moving Avg, window=5)')

    # 畫檔案分界線（最後一個不畫，那是結尾）
    for ep_end, fname in boundaries[:-1]:
        plt.axvline(x=ep_end, color='gray', linestyle='--', alpha=0.6)
        plt.text(ep_end + 0.3, max(all_penalties) * 0.98, fname[:20],
                 fontsize=7, color='gray', rotation=90, va='top')

    if final_data_mode == "episode":
        x_label = "Episode"
        y_label = "Total Delay Penalty per Episode"
        title_suffix = f"{len(all_penalties)} episodes"
    else:
        x_label = "Decision Step (every 10s)"
        y_label = "Delay Penalty per Step"
        title_suffix = f"{len(all_penalties)} steps (test mode)"
    plt.title(f'RL Learning Curve — Model: {model_id} ({title_suffix})')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    timestamp = get_timestamp()
    output_name = f"RL_Convergence_{model_id}_{timestamp}.png"
    plt.savefig(output_name, dpi=150)
    plt.close()
    print(f"✅ RL 收斂圖已儲存為: {output_name}")


# ==============================================================================
# GA 模式
# ==============================================================================

def plot_ga_single(csv_files):
    """
    single 模式：每個 GA CSV 獨立畫一張收斂圖。
    支援新版（total_score 欄位）與舊版（delay 欄位）CSV 格式。
    """
    for filepath in csv_files:
        if not os.path.exists(filepath):
            print(f"⚠️ 找不到檔案，跳過: {filepath}")
            continue
        print(f"📊 [single] 處理: {os.path.basename(filepath)}")
        generations, best_scores = extract_ga_data(filepath)

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_scores, marker='o', linestyle='-',
                 color='blue', label='Best Score per Generation')
        plt.title(f'GA Convergence: {os.path.basename(filepath)}')
        plt.xlabel('Generation')
        plt.ylabel('Total Score (Penalty, lower is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        timestamp = get_timestamp()
        base = os.path.basename(filepath).replace('.csv', '')
        output_name = f"GA_Convergence_{base}_{timestamp}.png"
        plt.savefig(output_name, dpi=150)
        plt.close()
        print(f"   → 已儲存: {output_name}")


def plot_ga_combine(csv_files):
    """
    combine 模式：把多個 GA 訓練 CSV 合併成單一收斂圖。
    因為 GA 訓練會把歷史最佳帶入下一次訓練（Warm Start），
    所以用「歷代全局最佳」的視角來呈現整體收斂趨勢。
    """
    if not csv_files:
        print("❌ 沒有找到任何 GA CSV 檔案。")
        return

    csv_files = sorted([f for f in csv_files if os.path.exists(f)])
    if not csv_files:
        print("❌ 所有指定檔案均不存在。")
        return

    print(f"📊 [combine] 合併 {len(csv_files)} 個 GA 訓練檔案...")

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    global_best_so_far = float('inf')
    x_offset = 0  # 讓 X 軸的代數連續

    for i, filepath in enumerate(csv_files):
        print(f"   讀取: {os.path.basename(filepath)}")
        generations, best_scores = extract_ga_data(filepath)

        # 資料為空（非 CSV 或欄位不符）直接跳過
        if not generations or not best_scores:
            print(f"   → 跳過（無有效數據）")
            continue

        # 計算「考慮歷史最佳後的有效最佳分數」
        effective_best = []
        for score in best_scores:
            global_best_so_far = min(global_best_so_far, score)
            effective_best.append(global_best_so_far)

        x = [x_offset + g for g in generations]
        label = os.path.basename(filepath).replace('.csv', '')[:30]
        color = colors[i % len(colors)]

        # 原始每代最佳（透明虛線）
        plt.plot(x, best_scores, alpha=0.3, color=color, linestyle='--')
        # 全局累計最佳（實線）
        plt.plot(x, effective_best, color=color, linewidth=2, label=label)

        # 畫訓練分界線（確保 x 不為空才畫）
        if i < len(csv_files) - 1 and x:
            plt.axvline(x=x[-1], color='gray', linestyle=':', alpha=0.5)

        x_offset += max(generations) + 1  # 下一個訓練的起始代數

    plt.title(f'GA Combined Convergence (solid = global best so far)')
    plt.xlabel('Generation (continuous across runs)')
    plt.ylabel('Total Score (Penalty, lower is better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8, loc='upper right')

    timestamp = get_timestamp()
    output_name = f"GA_Combined_{timestamp}.png"
    plt.savefig(output_name, dpi=150)
    plt.close()
    print(f"✅ GA 合併收斂圖已儲存為: {output_name}")


# ==============================================================================
# Hybrid 模式
# ==============================================================================

def is_ga_file(filepath):
    if "GA" in filepath.upper() or filepath.endswith('.csv'):
        return True
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(2000)
            if "Score:" in sample or "generation" in sample:
                return True
    except:
        pass
    return False


def plot_hybrid(file1, file2):
    if is_ga_file(file1):
        ga_file, rl_file = file1, file2
    else:
        ga_file, rl_file = file2, file1

    print(f"📊 [hybrid] GA: {ga_file}")
    print(f"📊 [hybrid] RL: {rl_file}")

    _, ga_scores = extract_ga_data(ga_file)
    rl_data, _ = extract_episode_penalties(open(rl_file, encoding="utf-8").read())

    plt.figure(figsize=(12, 7))
    plt.plot(range(len(ga_scores)), ga_scores, label='GA Best (Generations)',
             color='blue', marker='s', markersize=4, alpha=0.8)
    plt.plot(range(len(rl_data)), rl_data, label='RL Raw (Episodes)',
             color='orange', alpha=0.3)

    rl_smooth = pd.Series(rl_data).rolling(window=5, min_periods=1).mean()
    plt.plot(range(len(rl_data)), rl_smooth, label='RL Trend (Moving Avg)',
             color='red', linewidth=2)

    plt.title('Algorithm Comparison: GA vs RL Convergence')
    plt.xlabel('Iteration (Generation for GA / Episode for RL)')
    plt.ylabel('Total Penalty / Delay')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    timestamp = get_timestamp()
    output_name = f"Hybrid_Comparison_{timestamp}.png"
    plt.savefig(output_name, dpi=150)
    plt.close()
    print(f"✅ 混合對比圖已儲存為: {output_name}")


# ==============================================================================
# 主程式
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="交通控制演算法收斂對比工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
使用範例：
  RL 模式（自動確認同一 model，支援萬用字元）：
    python3 plot_convergence.py RL execute_RL_2026040*

  GA 單獨模式（每個 CSV 各自一張圖）：
    python3 plot_convergence.py GA single execute_GA_*.csv

  GA 合併模式（多次訓練合併為單一收斂圖）：
    python3 plot_convergence.py GA combine execute_GA_*.csv

  Hybrid 模式：
    python3 plot_convergence.py hybrid ga_file.csv rl_file.txt
        """
    )
    parser.add_argument("mode", choices=["GA", "RL", "hybrid"], help="執行模式")
    parser.add_argument("files", nargs='*',
                        help="GA 模式：第一個參數為 combine 或 single，後面接檔案\n"
                             "RL 模式：直接接一或多個 log 檔案\n"
                             "hybrid 模式：一個 GA 檔 + 一個 RL 檔")

    args = parser.parse_args()

    if args.mode == "hybrid":
        if len(args.files) != 2:
            print("❌ hybrid 模式需要剛好 2 個檔案")
            return
        plot_hybrid(args.files[0], args.files[1])

    elif args.mode == "RL":
        plot_rl(args.files)

    elif args.mode == "GA":
        if len(args.files) == 0 or (len(args.files) == 1 and args.files[0].lower() in ("combine", "single")):
            print("📖 GA 模式使用方式：")
            print("   單獨模式（每個檔案各畫一張圖）：")
            print("     python3 plot_convergence.py GA single execute_GA_*.csv")
            print("     python3 plot_convergence.py GA single execute_GA_*.txt")
            print("   合併模式（多次訓練合併成單一收斂圖）：")
            print("     python3 plot_convergence.py GA combine execute_GA_*.csv")
            print("     python3 plot_convergence.py GA combine execute_GA_*.txt")
            return
        submode = args.files[0].lower()
        input_files = args.files[1:]
        if submode == "single":
            plot_ga_single(input_files)
        elif submode == "combine":
            plot_ga_combine(input_files)
        else:
            print(f"❌ 未知的 GA 子命令: [{submode}]")
            print("   請使用 combine 或 single")
            print("   範例: python3 plot_convergence.py GA combine execute_GA_*.csv")


if __name__ == "__main__":
    main()