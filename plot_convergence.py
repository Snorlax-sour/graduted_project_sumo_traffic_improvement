import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime  # 👑 新增：用於產生時間戳記

def get_timestamp():
    """產生目前的 YYYYMMDD_HHMM 格式字串"""
    return datetime.now().strftime("%Y%m%d_%H%M")



def plot_rl(txt_file):
    """
    處理 RL 訓練日誌並存檔
    """
    print(f"🧠 正在處理 RL 數據: {txt_file}")
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()

    episodes = re.split(r"🏁 正在啟動第 \d+ 局", content)[1:]
    episode_penalties = []
    
    for ep_content in episodes:
        delays = re.findall(r"延遲罰:\s*([\d.]+)", ep_content)
        if delays:
            total_delay = sum(float(d) for d in delays)
            episode_penalties.append(total_delay)

    if not episode_penalties:
        print("❌ 找不到可用的 RL 訓練數據。")
        return

    plt.figure(figsize=(10, 6))
    x = range(1, len(episode_penalties) + 1)
    plt.plot(x, episode_penalties, alpha=0.3, color='orange', label='Episode Total Delay')
    series = pd.Series(episode_penalties)
    smooth = series.rolling(window=5, min_periods=1).mean()
    plt.plot(x, smooth, color='red', linewidth=2, label='Trend (Moving Avg)')
    
    plt.title(f'RL Learning Curve: {os.path.basename(txt_file)}')
    plt.xlabel('Episode')
    plt.ylabel('Total Delay per Episode')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 👑 修改檔名：加入時間戳記
    timestamp = get_timestamp()
    output_name = f"RL_Convergence_{timestamp}.png"
    
    plt.savefig(output_name)
    plt.close()
    print(f"✅ RL 收斂圖已儲存為: {output_name}")


def is_ga_file(filepath):
    """偵測是否為 GA 檔案 (檢查檔名或內容)"""
    if "GA" in filepath.upper() or filepath.endswith('.csv'):
        return True
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(2000)
            if "Score:" in sample or "generation" in sample:
                return True
    except: pass
    return False

def extract_ga_data(file):
    """從 CSV 或 TXT 提取 GA 最佳分數"""
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        # 按照代數分組，取每一代最優個體的分數
        gen_best = df.groupby('generation')['total_score'].min().values
        return gen_best
    else:
        # 從 TXT 提取 (假設每代 30 個體)
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        scores = [float(x) for x in re.findall(r"Score:\s*([\d.]+)", content)]
        # 每 30 個 Score 取一個最小值
        return [min(scores[i:i+30]) for i in range(0, len(scores), 30) if i+30 <= len(scores)]

def extract_rl_data(file):
    """從 TXT 提取 RL 每局總延遲"""
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
    episodes = re.split(r"🏁 正在啟動第 \d+ 局", content)[1:]
    episode_penalties = []
    for ep_content in episodes:
        delays = re.findall(r"延遲罰:\s*([\d.]+)", ep_content)
        if delays:
            episode_penalties.append(sum(float(d) for d in delays))
    return episode_penalties

def plot_ga(input_file):
    """
    處理 GA 檔案並存檔 (智慧支援 CSV 與 TXT)
    """
    print(f"📊 正在處理 GA 數據: {input_file}")
    
    # 👑 智慧判斷：如果是 CSV 就用 pandas，如果是 TXT 就用正則表達式提取
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        generations = df.groupby('generation')['total_score'].min().index
        best_scores = df.groupby('generation')['total_score'].min().values
    else:
        best_scores = extract_ga_data(input_file)
        generations = range(len(best_scores))
        
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_scores, marker='o', linestyle='-', color='blue', label='Best Score')
    
    plt.title(f'GA Convergence Plot: {os.path.basename(input_file)}')
    plt.xlabel('Generation')
    plt.ylabel('Total Score (Penalty Value)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    timestamp = get_timestamp()
    output_name = f"GA_Convergence_{timestamp}.png"
    
    plt.savefig(output_name)
    plt.close() # 關閉畫布節省記憶體
    print(f"✅ GA 收斂圖已儲存為: {output_name}")


def plot_hybrid(file1, file2):
    """混合模式繪圖"""
    # 偵測檔案類型
    if is_ga_file(file1):
        ga_file, rl_file = file1, file2
    else:
        ga_file, rl_file = file2, file1

    print(f"📊 [混合模式] 偵測到 GA 檔案: {ga_file}")
    print(f"📊 [混合模式] 偵測到 RL 檔案: {rl_file}")

    ga_data = extract_ga_data(ga_file)
    rl_data = extract_rl_data(rl_file)

    plt.figure(figsize=(12, 7))
    
    # 繪製 GA 曲線
    plt.plot(range(len(ga_data)), ga_data, label='GA Best (Generations)', color='blue', marker='s', markersize=4, alpha=0.8)
    
    # 繪製 RL 曲線
    plt.plot(range(len(rl_data)), rl_data, label='RL Raw (Episodes)', color='orange', alpha=0.3)
    rl_series = pd.Series(rl_data)
    rl_smooth = rl_series.rolling(window=5, min_periods=1).mean()
    plt.plot(range(len(rl_data)), rl_smooth, label='RL Trend (Moving Avg)', color='red', linewidth=2)

    plt.title('Algorithm Comparison: GA vs RL Convergence')
    plt.xlabel('Iteration (Generation for GA / Episode for RL)')
    plt.ylabel('Total Penalty / Delay')
    plt.yscale('log')  
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    timestamp = get_timestamp()
    output_name = f"Hybrid_Comparison_{timestamp}.png"
    
    # 👑 修復存檔 Bug：必須先 savefig，再 close，並且移除 show() 避免卡住終端機
    plt.savefig(output_name)
    plt.close()
    print(f"✅ 混合對比圖已儲存為: {output_name}")

def main():
    parser = argparse.ArgumentParser(description="交通控制演算法收斂對比工具")
    parser.add_argument("mode", choices=["GA", "RL", "hybrid"], help="執行模式")
    parser.add_argument("files", nargs='+', help="輸入檔案")

    args = parser.parse_args()

    if args.mode == "hybrid":
        if len(args.files) != 2:
            print("❌ 錯誤：混合模式需要提供兩個檔案 (一個 GA, 一個 RL)")
            return
        plot_hybrid(args.files[0], args.files[1])
    elif args.mode == "GA":
        plot_ga(args.files[0]) # 沿用之前的單一繪圖邏輯
    elif args.mode == "RL":
        plot_rl(args.files[0])

if __name__ == "__main__":
    main()