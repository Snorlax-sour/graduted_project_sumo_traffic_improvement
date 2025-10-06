import re
import matplotlib.pyplot as plt

def plot_log_data(log_file='execute.txt'):
    """
    讀取並解析指定的日誌檔案，然後繪製訓練數據圖表。
    """
    # 初始化用於存放數據的列表
    steps = []
    rewards = []
    epsilons = []

    # 定義解析日誌行的規則 (正規表示式)
    # 它會尋找 "時間: ...s | 獎勵: ... | Epsilon: ..." 的格式
    pattern = re.compile(r"時間: (\d+)s \| 獎勵: (-?\d+\.\d+) \| Epsilon: (\d+\.\d+)")

    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    # 如果成功解析，就提取數字並存入列表
                    steps.append(int(match.group(1)))
                    rewards.append(float(match.group(2)))
                    epsilons.append(float(match.group(3)))
        
        if not steps:
            print(f"在 '{log_file}' 中找不到任何可供繪圖的數據。")
            return

        # --- 開始繪圖 ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # 圖一：獎勵變化曲線
        ax1.plot(steps, rewards, label='每 10 步的獎勵', color='deepskyblue')
        ax1.set_ylabel('獎勵值 (Reward)')
        ax1.set_title('強化學習訓練過程分析')
        ax1.legend()
        ax1.grid(True)

        # 計算移動平均獎勵，讓趨勢更明顯
        window_size = 50 # 每 50 個數據點計算一次平均
        if len(rewards) >= window_size:
            moving_avg = [sum(rewards[i-window_size:i]) / window_size for i in range(window_size, len(rewards))]
            ax1.plot(steps[window_size-1:], moving_avg, label=f'{window_size}步移動平均獎勵', color='red', linewidth=2)
            ax1.legend()


        # 圖二：探索率 (Epsilon) 衰減曲線
        ax2.plot(steps, epsilons, label='探索率 (Epsilon)', color='orange')
        ax2.set_xlabel('模擬時間步 (Simulation Steps)')
        ax2.set_ylabel('探索率 (Epsilon)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_results.png') # 將圖表儲存成圖片
        print("圖表已成功繪製並儲存為 'training_results.png'")
        plt.show()

    except FileNotFoundError:
        print(f"錯誤：找不到日誌檔案 '{log_file}'。請確認檔案名稱和路徑是否正確。")

if __name__ == '__main__':
    # 當您執行這個腳本時，它會自動尋找 'execute.txt'
    plot_log_data()