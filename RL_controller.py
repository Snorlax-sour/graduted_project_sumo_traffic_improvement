import traci
import sys
import os
from DQN_RL_Agent import DQNAgent # 直接 import class
import csv # <--- 新增
GA_RESULT_PATH = "./GA_best_result.csv"
last_total_waiting_time = 0.0

def read_ga_optimal_phases(csv_filepath):
    """從 GA 輸出的 CSV 檔案中讀取最後一行的 phase1 和 phase2 數值"""
    
    # 預設經驗值 (如果找不到檔案，則用經驗值代替，避免程式崩潰)
    DEFAULT_PHASES = [35.0, 25.0] 

    if not os.path.exists(csv_filepath):
        print(f"警告：找不到 GA 結果檔案 '{csv_filepath}'，使用預設經驗值 {DEFAULT_PHASES} 作為靜態基線。")
        return DEFAULT_PHASES

    try:
        with open(csv_filepath, 'r') as f:
            reader = csv.reader(f)
            # 確保檔案不為空
            rows = list(reader)
            if len(rows) < 2: # 至少要有標頭和一行數據
                print(f"警告：'{csv_filepath}' 檔案中數據不足，使用預設經驗值 {DEFAULT_PHASES}。")
                return DEFAULT_PHASES

            header = rows[0]
            last_row = rows[-1]
            
            # 找到 phase1 和 phase2 在 CSV 中的欄位索引
            phase1_index = header.index('phase1')
            phase2_index = header.index('phase2')
            
            # 讀取數值並轉為 float
            phase1 = float(last_row[phase1_index])
            phase2 = float(last_row[phase2_index])
            print(f"✅ 成功從 '{csv_filepath}' 讀取 GA 最佳解：[{phase1}, {phase2}]")
            return [phase1, phase2]
            
    except Exception as e:
        # 捕獲所有可能的錯誤，例如格式錯誤、欄位找不到等
        print(f"讀取 GA 結果時發生嚴重錯誤 ({e})，使用預設經驗值 {DEFAULT_PHASES}。")
        return DEFAULT_PHASES
    
# 【DQN/GA 混合策略的核心數據輸入】
# 在程式啟動時呼叫函數，動態載入最佳時相
GA_OPTIMAL_PHASES = read_ga_optimal_phases(GA_RESULT_PATH)

def get_sumo_home():
    """找到 SUMO 的安装目录并設定環境"""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        return True
    else:
        sys.exit("請確認 SUMO_HOME 環境變數已設定！")

def get_state(tls_id):
    """
    【修復】: 獲取指定交通號誌的狀態，納入時相和 GA 最佳解。
    """
    
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    unique_lanes = list(set(lanes))
    
    # 1. 獲取排隊長度 (Queue Lengths)
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes]
    
    # 2. 獲取當前時相 (Current Phase) 
    current_phase = traci.trafficlight.getPhase(tls_id)
    
    # 3. 異構集成 (RL+GA) - 現在使用動態讀取的數值
    GA_min_time_suggestion = 0.0
    
    # 假設 Phase 0 是主幹道綠燈，Phase 2 是次幹道綠燈
    if current_phase == 0:  
        GA_min_time_suggestion = GA_OPTIMAL_PHASES[0] # 主幹道最佳時長
    elif current_phase == 2: 
        GA_min_time_suggestion = GA_OPTIMAL_PHASES[1] # 次幹道最佳時長
    
    # 4. 組合狀態：將所有數值扁平化為一個 tuple
    state_list = queue_lengths + [current_phase] + [GA_min_time_suggestion]
    
    return tuple(state_list)
# -----------------------------
    
# context: 在程式的全局區域
# --- 修正後的 calculate_reward 函數 ---
def calculate_reward(tls_id):
    """
    計算即時獎勵：負的總等待時間。
    
    traci.lane.getWaitingTime(lane) 會返回在最近的模擬步驟中，
    車輛在該車道上等待的累積時間（單位：秒）。
    這是一個即時懲罰，非常適合 RL 訓練。
    """
    try:
        # 1. 獲取所有受該路口控制的車道
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        
        # 2. 計算所有車道的總等待時間
        total_waiting_time = 0.0
        # 遍歷所有受控車道，計算它們在當前步驟中的總等待時間
        # 注意：這個值通常在每個 time step 後會被重置，
        # 或者指在當前 time step 期間，車輛等待的累積時間。
        # (在 SUMO 中，它是指當前在該車道上等待的車輛的總累積等待時間，是一個「狀態」指標)
        for lane in lanes:
            # getWaitingTime: 返回在當前 time step 期間，車輛在車道上等待的累積時間。
            # getAccumulatedWaitingTime: 返回自上次重置以來，總累積等待時間。
            # 為了即時獎勵，使用 getWaitingTime 較為合適。
            total_waiting_time += traci.lane.getWaitingTime(lane)
            
        # 3. 定義獎勵：最小化等待時間 (負的等待時間)
        reward = -total_waiting_time
        
        # 第二個回傳值 (例如用於統計)
        return reward, total_waiting_time
        
    except traci.TraCIException as e:
        # 如果路口 ID 錯誤或 TraCI 連線中斷
        # print(f"計算獎勵時發生 TraCI 錯誤: {e}", file=sys.stderr)
        return 0.0, 0.0 # 回傳 0 獎勵以避免崩潰
    

def run_experiment():
    """主模擬迴圈"""
    
    # --- 1. 初始化設定 ---
    SUMO_BINARY = "sumo" # 開發時用 sumo-gui, 訓練時用 sumo
    SUMOCFG_PATH = "./osm.sumocfg"
    SUMO_CMD = [SUMO_BINARY, "-c", SUMOCFG_PATH, "--quit-on-end", "--time-to-teleport", "-1"]
    
    TRAFFIC_LIGHT_ID = "1253678773"
    ACTION_SPACE = [0, 1]
    MIN_GREEN_TIME = 10
    DECISION_INTERVAL = 5
    
    # --- 2. 啟動 SUMO 並獲取初始資訊 ---
    try:
        traci.start(SUMO_CMD)
        
        logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
        num_phases = len(logics[0].phases) if logics else 4
        print(f"成功獲取交通號誌 '{TRAFFIC_LIGHT_ID}' 的相位總數: {num_phases}")

        lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
        state_size = len(list(set(lanes))) + 2
        print(f"狀態維度 (State Size): {state_size}")
        
    except traci.TraCIException as e:
        print(f"啟動SUMO或獲取資訊時出錯: {e}")
        traci.close()
        return

    # --- 3. 初始化 Agent ---
    agent = DQNAgent(state_size=state_size, action_space=ACTION_SPACE)

    # --- 4. 主模擬與學習迴圈 ---
    # context: 在 run_experiment 函式的內部
    
    # --- 4. 主模擬與學習迴圈 ---
    step = 0
    last_arrived_vehicles = 0 # 追踪上一轮的到达车辆数

    while step < 5000:
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("所有車輛已離開模擬，提前結束。")
                break

            current_state = get_state(TRAFFIC_LIGHT_ID)
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            # 獲取當前相位的總時間長度（綠燈時間 + 黃燈時間，通常是 3s）
            phase_duration = traci.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID)
            # 獲取當前相位已經持續的時間
            time_elapsed = traci.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID)
            # 獲取當前相位模式（例如：rryyGGggrr...）
            phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
            
            action = 0
            if current_phase % 2 == 0 and step > MIN_GREEN_TIME: # 稍微调整判断逻辑
                action = agent.choose_action(current_state)

            if action == 1 and current_phase % 2 == 0:
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
                
                # 等待黃燈時間
                for _ in range(3):
                    traci.simulationStep()
                    
                    step += 1
            else:
                # (修正点): 如果决定维持，就让模拟继续跑 DECISION_INTERVAL 步
                for _ in range(DECISION_INTERVAL):
                # 在进入下一步之前，先检查模拟是否已经没有车辆
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break # 如果没有车了，就跳出这个小循环
                    traci.simulationStep()
                    step += 1
                
            # --- 學習步驟 ---
            next_state = get_state(TRAFFIC_LIGHT_ID)
           
            reward, _ = calculate_reward(TRAFFIC_LIGHT_ID)
            
            agent.learn(current_state, action, reward, next_state)

            if step > 0:
                    
                print(f"時間: {step}s | 獎勵: {reward:.2f} | Epsilon: {agent.exploration_rate:.3f}", flush=True)
                print(f"  > TL 狀態: Phase Index={current_phase}, State='{phase_state}', Duration={phase_duration:.1f}s, Time Elapsed={time_elapsed:.1f}s", flush=True)
        except traci.TraCIException:
            print("SUMO 連線中斷，提前結束迴圈。")
            break
    # --- 5. 結束模擬 ---
    print("正在關閉模擬...")
    traci.close()

# --- 程式進入點 ---
if __name__ == "__main__":
    get_sumo_home()
    run_experiment()
    print("程式執行完畢！")