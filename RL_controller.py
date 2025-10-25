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
    # --- 【修正點 A：傳遞 ID 給 DQNAgent】---
    # 注意：RL_INSTANCE_ID 必須在 RL_controller.py 的檔案開頭從 sys.argv 讀取
    RL_INSTANCE_ID = "default_rl_instance" # 提供一個預設值
    if len(sys.argv) > 1:
        # 使用 sys.argv[1] 作為最常見的單參數傳遞方式
        RL_INSTANCE_ID = sys.argv[1] 
    print(f"使用的 RL 實例 ID (instance_id): {RL_INSTANCE_ID}")
    agent = DQNAgent(state_size=state_size, action_space=ACTION_SPACE, instance_id=RL_INSTANCE_ID)
    # 嘗試載入模型，如果存在的話
    agent.load_model() # <--- 新增：嘗試載入模型
    # --- 4. 主模擬與學習迴圈 ---
    # context: 在 run_experiment 函式的內部
    
    # --- 4. 主模擬與學習迴圈 ---
    step = 0
    # last_arrived_vehicles = 0 # 追踪上一轮的到达车辆数

    while step < 5000:
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("所有車輛已離開模擬，提前結束。")
                break

            current_state = get_state(TRAFFIC_LIGHT_ID)
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            # 獲取當前相位的總時間長度（綠燈時間 + 黃燈時間，通常是 3s）
            phase_duration = traci.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID)
           # 獲取距離下次變換剩餘的時間 (Time to Next Switch)
            # 這樣比嘗試計算 elapsed time 更準確且常見
            time_remaining = traci.trafficlight.getNextSwitch(TRAFFIC_LIGHT_ID) - traci.simulation.getTime()
            # 獲取當前相位模式（例如：rryyGGggrr...）
            phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
            
            action = 0
            # 只有在綠燈相位 (偶數相位) 且超過最短綠燈時間後才允許 RL 決策
            if current_phase % 2 == 0 and time_remaining <= (phase_duration - MIN_GREEN_TIME - 3): # 稍微调整判断逻辑
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
                print(f"  > TL 狀態: Phase Index={current_phase}, State='{phase_state}', Duration={phase_duration:.1f}s, Time Remaining={time_remaining:.1f}s", flush=True)
        except traci.TraCIException:
            print("SUMO 連線中斷，提前結束迴圈。")
            break
    # --- 5. 結束模擬 ---
    print("正在關閉模擬...")
    # --- 【修正點 B：儲存最終模型】---
    # 【修正點 4：儲存最終模型】
    agent.save_model() # <--- 新增：儲存模型
    traci.close()

# --- 1. 新增命令列/互動式參數解析函數 ---
def parse_arguments():
    """解析命令行參數，允許使用者選擇模式和模型名稱。"""
    
    # 檢查是否有足夠的命令行參數 (例如: python RL_controller.py test my_model)
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ['train', 'test']:
            print("❌ 錯誤: 第一個參數必須是 'train' 或 'test'。")
            sys.exit(1)
        
        # 第二個參數是模型名稱，如果沒有提供，則使用 'default_rl'
        instance_id = sys.argv[2] if len(sys.argv) > 2 else "default_rl"
        return mode, instance_id
    
    # 互動式模式：如果沒有命令行參數，則進入互動模式
    print("\n--- DQN 模式選擇 ---")
    while True:
        mode_input = input("請選擇模式 (輸入 'train' 或 'test'): ").strip().lower()
        if mode_input in ['train', 'test']:
            mode = mode_input
            break
        else:
            print("輸入無效，請輸入 'train' 或 'test'。")
            
    instance_id = input(f"請輸入模型名稱 (Model ID/Instance ID, 預設: default_rl): ").strip()
    if not instance_id:
        instance_id = "default_rl"
        
    return mode, instance_id


def main():
    mode, instance_id = parse_arguments()
    is_train_mode = (mode == 'train')

    # 參數設定
    TRAFFIC_LIGHT_ID = "1253678773"
    SUMO_CONFIG_FILE = "osm.sumocfg"
    MAX_SIMULATION_STEPS = 5000 # 模擬總步數
    DECISION_INTERVAL = 5 # 每隔 5 步進行一次決策
    STATE_SIZE = 6
    ACTION_SPACE = [0, 1]  # 0: Maintain, 1: Change Phase
    
    # --- 2. 初始化 DQN 代理，使用解析出的 instance_id ---
    print(f"使用的 RL 實例 ID (instance_id): {instance_id}")
    agent = DQNAgent(STATE_SIZE, ACTION_SPACE, instance_id=instance_id)


    if is_train_mode:
        print("💡 模式：DQN 訓練模式 (Train Mode)。")
        # 載入 GA 基線數據 (訓練用)
        read_ga_optimal_phases(GA_RESULT_PATH)
        # 訓練模式會繼續 Epsilon 衰減 (可以選擇載入上次進度)
        if agent.load_model():
            print("✅ 找到上次訓練模型，將繼續訓練。")
        else:
            print("⚠️ 未找到模型檔案，將從頭開始訓練。")
            
    else: # 測試模式
        print("💡 模式：DQN 測試模式 (Test Mode)。")
        # --- 測試模式核心邏輯 ---
        if not agent.load_model():
            print(f"\n❌ 警告：測試模式下未能找到已訓練的模型檔案 (ID: {instance_id})，請先執行訓練。")
            sys.exit(1) # 測試模式下找不到模型就退出
            
        agent.exploration_rate = 0.0 # 鎖定探索率為 0，只執行利用(Exploitation)
        print(f"✅ 模型載入成功。探索率 Epsilon 已鎖定為 {agent.exploration_rate}。")
        
        
    # ... [啟動 SUMO 和 TraCI 連線]
    if not get_sumo_home():
        sys.exit(1)
        
    # 注意：測試模式下可能希望使用 'sumo-gui'
    # sumoCmd = ["sumo-gui" if not is_train_mode else "sumo", ...] 
    # 這裡為簡潔保持使用 sumo-gui，但訓練時應改為 "sumo" 提高速度
    sumoCmd = [
        "sumo-gui", # 為了測試方便，暫時用 GUI
        "-c", SUMO_CONFIG_FILE,
        "--time-to-teleport", "-1",
        "--tripinfo-output", "tripinfo.xml" 
    ]
    traci.start(sumoCmd)
    
    
    # --- 3. 初始化並開始模擬 ---
    step = 0
    total_reward = 0.0 
    logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
    num_phases = len(logics[0].phases) if logics else 4
    print(f"成功獲取交通號誌 '{TRAFFIC_LIGHT_ID}' 的相位總數: {num_phases}")
    print(f"狀態維度 (State Size): {STATE_SIZE}")

    # --- 4. 主模擬與訓練/測試迴圈 ---
    while step < MAX_SIMULATION_STEPS:
        try:
            # --- 狀態獲取與動作選擇 ---
            current_state = get_state(TRAFFIC_LIGHT_ID)
            # 使用訓練/測試模式下的 Epsilon 選擇動作
            action = agent.choose_action(current_state) 

            # ... [省略動作執行代碼 (維持/切換黃燈/切換綠燈)]
            # 確保這裡的 set_traffic_light_phases 和 simulationStep 邏輯正確執行
            # ...
            
            # --- 執行動作的完整步驟 (範例，請替換成您的實際邏輯) ---
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            phase_state = traci.trafficlight.getPhaseState(TRAFFIC_LIGHT_ID)
            phase_duration = traci.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID)
            
            if action == 1: # 切換到下一相位 (需要黃燈緩衝)
                # 執行黃燈
                # ... (執行 set_traffic_light_phases, traci.simulationStep)
                pass # 這裡應包含您原始的黃燈切換邏輯
            else: # 維持當前相位
                # 執行 DECISION_INTERVAL 步
                for _ in range(DECISION_INTERVAL):
                    traci.simulationStep()
                    step += 1
            
            # --- 學習步驟 (僅限訓練模式) ---
            next_state = get_state(TRAFFIC_LIGHT_ID)
            reward, total_waiting_time = calculate_reward(TRAFFIC_LIGHT_ID)
            
            if is_train_mode:
                agent.learn(current_state, action, reward, next_state)
            
            total_reward += reward # 累加總獎勵

            # --- 紀錄與輸出 ---
            if step > 0:
                # 重新獲取當前狀態資訊
                current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                phase_state = traci.trafficlight.getPhaseState(TRAFFIC_LIGHT_ID)
                phase_duration = traci.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID)
                time_elapsed = traci.trafficlight.getPhaseTime(TRAFFIC_LIGHT_ID) 
                
                # 根據模式決定輸出內容
                if is_train_mode:
                    status_line = f"時間: {step}s | 累積獎勵: {total_reward:.2f} | Epsilon: {agent.exploration_rate:.3f}"
                else:
                    status_line = f"時間: {step}s | 瞬間獎勵: {reward:.2f} | 總等待: {total_waiting_time:.2f}"
                
                print(status_line, flush=True)
                print(f"  > TL 狀態: Phase Index={current_phase}, State='{phase_state}', Duration={phase_duration:.1f}s, Time Elapsed={time_elapsed:.1f}s", flush=True)

        except traci.TraCIException:
            print("SUMO 連線中斷，提前結束迴圈。")
            break
            
    # --- 5. 結束模擬 ---
    print("正在關閉模擬...")
    traci.close()
    
    if is_train_mode:
        agent.save_model() # 訓練結束時儲存模型
    else:
        # 測試模式下的最終結果輸出
        print(f"\n✅ 測試完成！使用的模型 ID: {instance_id}")
        print(f"模擬總步數: {step}")
        print(f"總累計等待時間 (負獎勵總和): {-total_reward:.2f} 秒")


# --- 程式進入點 ---
if __name__ == "__main__":
    get_sumo_home()
    run_experiment()
    # --- 6. 命令行參數處理：新增一個參數來控制模式 ---
    # main()
    print("程式執行完畢！")