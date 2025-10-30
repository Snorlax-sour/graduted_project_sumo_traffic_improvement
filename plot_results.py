import re
import matplotlib.pyplot as plt
import numpy as np # <--- 新增 import numpy

def plot_log_data(log_file='execute.txt'):
    """
    讀取並解析指定的日誌檔案，然後繪製訓練數據圖表。
    """
    steps = []
    rewards = []
    epsilons = []

    pattern = re.compile(r"時間: (\d+)s \| 獎勵: (-?\d+\.\d+) \| Epsilon: (\d+\.\d+)")

    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    steps.append(int(match.group(1)))
                    rewards.append(float(match.group(2)))
                    epsilons.append(float(match.group(3)))
        
        if not steps:
            print(f"在 '{log_file}' 中找不到任何可供繪圖的數據。")
            return

        # --- 開始繪圖 ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # 圖一：獎勵變化曲線
        ax1.plot(steps, rewards, label='Reward per Decision Step', color='deepskyblue', alpha=0.5)
        # ax1.plot(steps, rewards, label='每步決策的獎勵', color='deepskyblue', alpha=0.5) 

        ax1.set_ylabel('Cumulative Reward')
        # ax1.set_ylabel('獎勵值 (Reward)')

        ax1.set_title('DQN Training Analysis')
        # ax1.set_title('強化學習訓練過程分析')
        
        # --- (修正点：使用 numpy 来进行更稳健的移动平均计算) ---
        window_size = 50 
        if len(rewards) >= window_size:
            # 使用 np.convolve 来计算移动平均，这是一种标准做法
            moving_avg_rewards = np.convolve(rewards, np.ones(window_size), 'valid') / window_size
            # 计算对应的 x 轴坐标
            moving_avg_steps = steps[window_size - 1:]
            
            # 确保长度一致才绘图
            if len(moving_avg_steps) == len(moving_avg_rewards):
                ax1.plot(moving_avg_steps, moving_avg_rewards, label=f'{window_size}-Step Moving Average Reward', color='red', linewidth=2)
            else:
                 print("警告: 移動平均計算後的數據長度不匹配，跳過繪製趨勢線。")
        if len(moving_avg_steps) == len(moving_avg_rewards):
            ax1.plot(moving_avg_steps, moving_avg_rewards, label=f'{window_size}-Step Moving Average Reward', color='red', linewidth=2)
            # ax1.plot(moving_avg_steps, moving_avg_rewards, label=f'{window_size}步移動平均獎勵', color='red', linewidth=2)
        ax1.legend()
        ax1.grid(True)
        
        # 圖二：探索率 (Epsilon) 衰減曲線
        ax2.plot(steps, epsilons, label='Exploration Rate (Epsilon)', color='orange')
        # ax2.plot(steps, epsilons, label='探索率 (Epsilon)', color='orange')

        ax2.set_xlabel('Simulation Step (s)')
        # ax2.set_xlabel('模擬時間 (s)')
        
        ax2.set_ylabel('Epsilon')
        # ax2.set_ylabel('Epsilon')

        ax2.set_title('Epsilon Decay')
        # ax2.set_title('探索率 (Epsilon) 衰減曲線')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_results.png')
        print("圖表已成功繪製並儲存為 'training_results.png'")
        plt.show()

    except FileNotFoundError:
        print(f"錯誤：找不到日誌檔案 '{log_file}'。")

if __name__ == '__main__':
    plot_log_data("execute_RL_202510260818.txt")    