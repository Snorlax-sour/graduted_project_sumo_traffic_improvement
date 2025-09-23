import traci
import sys
import os
from RL_Agent import RLAgent
import random # 您的 RLAgent 會用到這個
"""
1. 初始化順序：將所有變數和函式的定義都移到 traci.start() 之前，
    這是程式設計的標準做法，確保在主迴圈開始前，所有「工具」都已經準備就緒。
2. 單一且正確的迴圈邏輯：強化學習的生命週期是 
    「觀察 -> 決策 -> 行動 -> 觀察結果 -> 學習」。
        這個循環必須在每一個時間步（或每一個決策點）內完整地發生。我將您兩段分離的程式碼合併到了 while 迴圈內部，並確保了這個核心循環的正確性。
3. 精確控制 traci.simulationStep()：在一個迴圈迭代中，通常只會有一個 
    traci.simulationStep()，它代表「時間前進一秒」。
        所有其他的操作都是在這一秒鐘前後發生的即時計算，
            這樣才能確保智能體學到的因果關係是正確的。
"""
# --- 設定 SUMO 環境 ---
# 確保您的 SUMO_HOME 環境變數已設置
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# --- 模擬設定 ---
# 選擇 sumo-gui 可以看到畫面，選擇 sumo 則是在背景運行，速度較快
sumo_binary = "sumo"  # (修改建議): 在開發階段建議使用 "sumo-gui" 方便觀察
# 指向您的 SUMO 設定檔
sumocfg_path = "./osm.sumocfg"  # 假設 .py 檔和 .sumocfg 在同一個資料夾

# 這是啟動 SUMO 的指令
# TraCI 會透過這個指令來開啟一個 SUMO 實例作為伺服器
sumo_cmd = [sumo_binary, "-c", sumocfg_path, "--quit-on-end"] # (修改): 加上 --quit-on-end 讓模擬結束後自動關閉


# --- 主程式 ---
def run_simulation():
    """主模擬迴圈"""

    # (修改點1：將所有變數設定、函式定義都移到迴圈開始前，確保它們被正確初始化)
    TRAFFIC_LIGHT_ID = "1253678773"  # <--- 您已經成功找到了正確的ID！
    # 透過osm.net.xml檔案來看有個tlLogic tag
    ACTION_SPACE = [0, 1]           # 0 = 維持, 1 = 切換

    # (修改點2：為了讓程式更有彈性，我們先啟動一次SUMO來獲取號誌的總相位數)
    try:
        traci.start([sumo_binary, "-c", sumocfg_path])
        # (修正): 使用正確的 TraCI 函式來獲取完整的相位定義
        logic = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
        if logic: # 確保成功獲取到邏輯
            num_phases = len(logic[0].phases)
            print(f"成功獲取交通號誌 '{TRAFFIC_LIGHT_ID}' 的相位總數: {num_phases}")
        else:
            # 如果找不到，給一個預設值並提示錯誤
            print(f"錯誤：找不到 ID 為 '{TRAFFIC_LIGHT_ID}' 的交通號誌邏輯，將使用預設值 4。")
            num_phases = 4 # 使用一個安全的預設值
        traci.close(wait=False) # <--- 在這裡關閉臨時連線

    except traci.TraCIException as e:
        print("請確認SUMO已安裝並在環境變數中，且 osm.sumocfg 檔案存在")
        print(f"啟動SUMO獲取資訊時出錯: {e}")
        return

    agent = RLAgent(action_space=ACTION_SPACE)

    def perform_action(action, current_phase):
        """執行動作"""
        if action == 1:
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            next_phase = (current_phase + 1) % num_phases
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
            
    def get_state():
        """獲取狀態: 將所有進入路口的車道排隊長度組合起來"""
        incoming_lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
        state_lanes = list(set(incoming_lanes)) 
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in state_lanes]
        return tuple(queue_lengths)
    
    def calculate_reward():
        """計算獎勵: 負的總等待時間"""
        incoming_lanes = list(set(traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)))
        total_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in incoming_lanes)
        return -total_waiting_time

    # 1. 啟動 SUMO
    # traci.start 會在背景執行上面的 sumo_cmd 指令，並與之建立連線
    traci.start(sumo_cmd)
    
    step = 0
    # 2. 模擬主迴圈
    # traci.simulation.getMinExpectedNumber() 的意思是：
    # "只要模擬中還有車輛 (不論是在路上跑的，還是未來規劃要出現的)，迴圈就繼續"
    while step < 5000: # 先設定一個最大模擬步數
        try: # <--- 在這裡加上 try
        # (修改點3：這是整個程式邏輯的核心修正)
        # 在每個時間步，RL Agent 都需要完整地走一次「觀察->決策->行動->學習」的流程
        
        # --- 在這裡，我們可以開始加入我們的 RL 邏輯 ---
            
            # I. 觀察當前環境狀態
            current_state = get_state()
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)

            action = 0 # 預設動作為「不切換」
            # II. 只有在綠燈相位(偶數位)，才讓Agent做決策
            if current_phase % 2 == 0:
                action = agent.choose_action(current_state)

            if action == 1:
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
            # III. 執行選擇的動作
            # perform_action(action, current_phase) <--- 為啥刪除了, 
            # ai studio gemini說將內聯了，簡易功能融入裡面，希望可以後續分離
            
            # IV. 讓模擬前進一步，以觀察執行動作後產生的結果
            # 讓模擬前進一個時間步 (預設是 1 秒)
            # --- 核心邏輯：整個迴圈只在這裡讓時間前進 ---
            traci.simulationStep()
            step += 1
            # V. 獲取新狀態並計算獎勵，然後讓Agent學習
            next_state = get_state()
            reward = calculate_reward()
            agent.learn(current_state, action, reward, next_state)
            
            # 印出學習過程中的資訊
            if step % 10 == 0: # 每10秒印一次資訊，避免洗版
                print(f"時間: {step}s | 獎勵: {reward:.2f} | Epsilon: {agent.exploration_rate:.3f}")
            
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("所有車輛已離開模擬，提前結束。")
                break

      
        except traci.TraCIException: # <--- 在迴圈的最後加上 except
            print("SUMO 連線中斷或模擬已結束，正在優雅地退出。")
            break # 跳出 while 迴圈
    # 3. 結束模擬
    print("正在關閉模擬...")
    traci.close()
    sys.stdout.flush()

# --- 程式進入點 ---
if __name__ == "__main__":
    run_simulation()
    print("模擬結束！")