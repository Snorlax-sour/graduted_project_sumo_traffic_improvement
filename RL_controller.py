import traci
import sys
import os
from DQN_RL_Agent import DQNAgent # 直接 import class
import csv # <--- 新增
from plyer import notification # <--- 新增

GA_RESULT_PATH = "./GA_best_result.csv"
last_total_waiting_time = 0.0
# --- 新增全域變數 ---
# context: 在 RL_controller.py 檔案的頂部新增/修改此行
last_total_queue_length = 0.0
last_total_cumulative_waiting_time = 0.0
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
# context: 在 RL_controller.py 檔案的頂部新增/修改此行
last_total_waiting_time = 0.0 # 重新啟用這個變數，並用來追蹤累積等待時間

def calculate_reward(tls_id):
    """
    計算即時獎勵：總累積等待時間的變化量 (Delta Delay)。
    使用 traci.vehicle.getWaitingTime 來實現截圖中的目標。
    """
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        
        current_total_waiting_time = 0.0
        
        # 1. 遍歷所有受控車道
        for lane in unique_lanes:
            # 獲取該車道上所有車輛 ID
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            
            # 2. 遍歷車道上的所有車輛
            for veh_id in vehicle_ids:
                # 【關鍵嘗試】：使用 traci.vehicle 級別的 API 來獲取單車等待時間
                # 這個 API 應該是相當穩定的
                current_total_waiting_time += traci.vehicle.getWaitingTime(veh_id)
        
        # 3. 計算 Delta Reward
        global last_total_waiting_time
        
        # Delta = (舊的累積等待時間 - 新的累積等待時間)
        # 如果 Delta > 0，表示等待時間減少，獎勵為正。
        delta_delay = last_total_waiting_time - current_total_waiting_time
        delta_delay *= -1
        # 🌟 【關鍵修正】：引入二次方排隊懲罰
    # 1. 獲取當前的總排隊車輛數 (Halting Number)
        current_total_queue_length  = get_total_queue_length(tls_id)
        # 4. 更新全域變數
        last_total_waiting_time = current_total_waiting_time
        # 2. 懲罰係數 (可調整 0.1 ~ 0.3)
        beta = 0.2 
        
        # 3. 最終獎勵 = 等待時間變化獎勵 - 二次方排隊懲罰
        reward =  delta_delay - (beta * (current_total_queue_length ** 2))
        # 5. 定義獎勵：最大化等待時間的減少
        
        # 第二個返回值 'current_total_waiting_time' 兼容主迴圈
        return reward, current_total_waiting_time
            
    except traci.TraCIException:
        # SUMO 連線中斷
        return 0.0, 0.0 
    except AttributeError as e:
        # 捕獲 'LaneDomain' 或 'VehicleDomain' 相關的 AttributeError
        print(f"計算獎勵時發生致命 AttributeError: {e}. 請檢查 traci.vehicle.getWaitingTime 是否存在。", file=sys.stderr)
        # 如果這個方法失敗，就回到我們之前最魯棒的「排隊長度變化量」邏輯
        return calculate_reward_queue_fallback(tls_id)
    except Exception as e_general:
        # 捕獲其他錯誤
        return 0.0, 0.0
def get_total_queue_length(tls_id):
    """
    計算給定交通號誌所控制的所有車道上的總排隊車輛數 (Halting Number)。
    """
    try:
        # 獲取交通號誌控制的所有車道
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        # 使用 set 去除重複的車道 ID (因為一個信號燈可能控制同一車道的不同部分)
        unique_lanes = list(set(lanes))
        
        total_queue_length = 0
        
        # 遍歷所有受控車道
        for lane in unique_lanes:
            # 獲取當前步驟靜止或排隊的車輛數 (速度小於 0.1 m/s)
            # traci.lane.getLastStepHaltingNumber(lane) 是計算排隊長度的標準 API
            total_queue_length += traci.lane.getLastStepHaltingNumber(lane)
            
        return total_queue_length
            
    except traci.TraCIException:
        # SUMO 連線中斷或其他 traci 錯誤
        return 0
    except Exception:
        # 其他錯誤
        return 0

# (注意：這個函式需要在您的程式碼中定義一次)
# --- 備用函數：如果 vehicle.getWaitingTime 失敗，則回退到排隊長度 ---
# 這是確保程式不會因為 API 不相容而崩潰的保護層
def calculate_reward_queue_fallback(tls_id):
    """
    備用函數：如果基於車輛等待時間的計算失敗，則回退到排隊長度變化量。
    （使用你上次修正後的 Delta Queue 邏輯）
    """
    print("使用calculate_reward_queue_fallback",flush=True)
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        
        current_total_queue_length = 0.0
        for lane in unique_lanes:
            current_total_queue_length += traci.lane.getLastStepHaltingNumber(lane)
        
        # 注意：為了避免依賴另一個全域變數，這裡暫時使用 last_total_waiting_time 作為排隊長度的追蹤器。
        # ⚠️ 這裡是犧牲了變數名稱的語義，換取程式的魯棒性。
        global last_total_waiting_time
        delta_queue = last_total_waiting_time - current_total_queue_length
        last_total_waiting_time = current_total_queue_length
        
        # 使用 Delta Queue 作為獎勵
        reward = delta_queue * 1.0 
        
        # 返回 排隊長度 (Queue Length)
        return reward, current_total_queue_length
            
    except traci.TraCIException:
        return 0.0, 0.0 
    except Exception as e_general:
        return 0.0, 0.0

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
        instance_id = sys.argv[2]
    elif len(sys.argv) == 2: # 處理只給了模式或ID的情況
        # 為了簡化，我們假設如果只有一個參數，它就是模型ID，並使用默認的訓練模式
        print(f"警告：只提供一個參數 '{sys.argv[1]}'，將其作為模型ID並進入默認的訓練模式。")
        instance_id = sys.argv[1]
        
    return mode, instance_id


# def main():
#     mode, instance_id = parse_arguments()
#     is_train_mode = (mode == 'train')

#     # 參數設定
#     TRAFFIC_LIGHT_ID = "1253678773"
#     SUMO_CONFIG_FILE = "osm.sumocfg"
#     MAX_SIMULATION_STEPS = 100000 # 模擬總步數
#     DECISION_INTERVAL = 5 # 每隔 5 步進行一次決策
#     MIN_GREEN_TIME = 10 # 最小綠燈時間
#     ACTION_SPACE = [0, 1]  # 0: Maintain, 1: Change Phase
    
#     # --- 2. 初始化 DQN 代理，使用解析出的 instance_id ---
#     print(f"使用的 RL 實例 ID (instance_id): {instance_id}")
#     agent = DQNAgent(state_size=6, action_space=ACTION_SPACE, instance_id=instance_id) # state_size 暫時為 0


#     if is_train_mode:
#         print("💡 模式：DQN 訓練模式 (Train Mode)。")
#         # 載入 GA 基線數據 (訓練用)
#         read_ga_optimal_phases(GA_RESULT_PATH)
#         # 訓練模式會繼續 Epsilon 衰減 (可以選擇載入上次進度)
#         if agent.load_model():
#             print("✅ 找到上次訓練模型，將繼續訓練。")
#         else:
#             print("⚠️ 未找到模型檔案，將從頭開始訓練。")
            
#     else: # 測試模式
#         print("💡 模式：DQN 測試模式 (Test Mode)。")
#         # --- 測試模式核心邏輯 ---
#         if not agent.load_model():
#             print(f"\n❌ 警告：測試模式下未能找到已訓練的模型檔案 (ID: {instance_id})，請先執行訓練。")
#             sys.exit(1) # 測試模式下找不到模型就退出
            
#         agent.exploration_rate = 0.0 # 鎖定探索率為 0，只執行利用(Exploitation)
#         print(f"✅ 模型載入成功。探索率 Epsilon 已鎖定為 {agent.exploration_rate}。")
        
        
#     # ... [啟動 SUMO 和 TraCI 連線]
#     if not get_sumo_home():
#         sys.exit(1)
#     # 決定使用的種子碼
#     # 訓練時使用固定種子 (例如 42)，測試時使用不同種子 (例如 100)
#     sim_seed = 42 if is_train_mode else 100 # <--- 這裡可以動態修改
#     # 【修正】: 根據模式自動選擇 sumo 或 sumo-gui
#     sumo_binary = "sumo-gui" if is_train_mode else "sumo-gui"
#     sumoCmd = [
#         sumo_binary,
#         "-c", SUMO_CONFIG_FILE,
#         "--time-to-teleport", "-1",
#         "--tripinfo-output", "tripinfo.xml" ,
#         "--seed", str(sim_seed) # 【新增】加入隨機種子碼
#     ]
#     traci.start(sumoCmd)
    
    
#     # --- 3. 初始化並開始模擬 ---
#     step = 0
#     cumulative_reward = 0.0
#     logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
#     num_phases = len(logics[0].phases) if logics else 4
#     # 【關鍵修正 1】：新增時間追蹤變數
#     time_since_last_change = 0
#     # 【修正】: 動態獲取 state_size 並建立模型
#     lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
#     real_state_size = len(list(set(lanes))) + 2
#     agent.state_size = real_state_size
#     agent.build_models() # 在獲取真實維度後，才建立模型

#     print(f"成功獲取交通號誌 '{TRAFFIC_LIGHT_ID}' 的相位總數: {num_phases}")
#     print(f"狀態維度 (State Size): {agent.state_size}")

#     # --- 4. 主模擬與訓練/測試迴圈 (最終穩定結構) ---
#     while step < MAX_SIMULATION_STEPS:
#         try:
#             # 1. 推進單步模擬與時間計數 (每一步都執行)
#             traci.simulationStep()
#             step += 1
#             time_since_last_change += 1 

#             # 2. 決策與學習邏輯：只在固定的 DECISION_INTERVAL 發生
#             if step % DECISION_INTERVAL == 0:
                
#                 current_state = get_state(TRAFFIC_LIGHT_ID)
#                 current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                
#                 action = 0 # 預設：維持
                
#                 # 2.1 必須是綠燈，且超過最小綠燈時間，才允許 Agent 決策切換
#                 if current_phase % 2 == 0 and time_since_last_change >= MIN_GREEN_TIME:
#                     action = agent.choose_action(current_state)
                    
#                 # 2.2 執行切換動作
#                 if action == 1:
#                     next_phase = (current_phase + 1) % num_phases
#                     traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
#                     time_since_last_change = 0 # 重置計時器
                
#                 # 2.3 學習與紀錄 (發生在每個決策點)
#                 next_state = get_state(TRAFFIC_LIGHT_ID)
#                 reward, current_total_queue_length = calculate_reward(TRAFFIC_LIGHT_ID)
                
#                 if is_train_mode:
#                     agent.learn(current_state, action, reward, next_state) 
                
#                 cumulative_reward += reward
#                 # 4. 執行 RL 動作 (切換相位)
#                 # 2.4 學習與紀錄 (發生在每個決策點)
#                 next_state = get_state(TRAFFIC_LIGHT_ID)
#                 reward, current_total_queue_length = calculate_reward(TRAFFIC_LIGHT_ID)
                
#                 if is_train_mode:
#                     agent.learn(current_state, action, reward, next_state)
                
#                 cumulative_reward += reward
                
#                 # 2.5 輸出紀錄
#                 time_info = f" | Phase Time: {time_since_last_change:.1f}s" # 這裡輸出的是新相位的運行時間
                
#                 if is_train_mode:
#                     status_line = f"時間: {step}s{time_info} | 獎勵: {reward:.2f} | Action: {action} | Epsilon: {agent.exploration_rate:.3f}"
#                 else:
#                     status_line = f"時間: {step}s{time_info} | 瞬間獎勵: {reward:.2f} | 排隊總數: {current_total_queue_length:.2f}"
                
#                 print(status_line, flush=True)

#             # 3. 處理非 RL 控制的相位跳轉 (黃燈 -> 紅燈/綠燈)
#             #    由於我們只在綠燈時才讓 Agent 決策切換，在黃燈/紅燈時，SUMO 會自動跳轉。
#             #    我們只需要確保在 SUMO 自動跳轉後，time_since_last_change 能正確反映新相位的運行時間。
#             #    (當 time_since_last_change 達到黃燈/紅燈的默認時長時，traci.getPhase() 會返回新值，
#             #     time_since_last_change 則從 1 開始累積，邏輯正確。)

#                 # --- 紀錄與輸出 ---
#                 if step > 0:
#                     # 使用正確的 time_since_last_change 進行輸出
#                     time_info = f" | 綠燈時間: {time_since_last_change:.1f}s"
                    
#                     if is_train_mode:
#                         status_line = f"時間: {step}s{time_info} | 獎勵: {reward:.2f} | Epsilon: {agent.exploration_rate:.3f}"
#                     else:
#                         # 測試模式下的 '總等待' 實際上是排隊總數 (current_total_queue_length)
#                         status_line = f"時間: {step}s{time_info} | 瞬間獎勵: {reward:.2f} | 排隊總數: {current_total_queue_length:.2f}"
                    
#                     print(status_line, flush=True)

#         except traci.TraCIException:
#             print("SUMO 連線中斷，提前結束迴圈。")
#             break
            
#     # --- 5. 結束模擬 ---
#     print("正在關閉模擬...")
#     traci.close()
    
#     if is_train_mode:
#         agent.save_model() # 訓練結束時儲存模型
#     else:
#         # 測試模式下的最終結果輸出
#         print(f"\n✅ 測試完成！使用的模型 ID: {instance_id}")
#         print(f"模擬總步數: {step}")
#         print(f"最終累積獎勵: {cumulative_reward:.2f}")
#     notification.notify(
#         title = "Python RL Trainning Finish",
#         message = f"RUN PID: {os.getpid()}, MODEL ID= {instance_id}" ,
            
#         # displaying time
#         timeout=100 # seconds
#     )   
def main():
    mode, instance_id = parse_arguments()
    is_train_mode = (mode == 'train')

    # 參數設定
    TRAFFIC_LIGHT_ID = "1253678773"
    SUMO_CONFIG_FILE = "osm.sumocfg"
    MAX_SIMULATION_STEPS = 25000 # 模擬總步數
    DECISION_INTERVAL = 5 # 每隔 5 步進行一次決策
    MIN_GREEN_TIME = 10 # 最小綠燈時間
    ACTION_SPACE = [0, 1]  # 0: Maintain, 1: Change Phase
    
    # --- 2. 初始化 DQN 代理，使用解析出的 instance_id ---
    print(f"使用的 RL 實例 ID (instance_id): {instance_id}")
    agent = DQNAgent(state_size=6, action_space=ACTION_SPACE, instance_id=instance_id) # state_size 暫時為 0


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
    # 決定使用的種子碼
    # 訓練時使用固定種子 (例如 42)，測試時使用不同種子 (例如 100)
    sim_seed = 42 if is_train_mode else 100 # <--- 這裡可以動態修改
    # 【修正】: 根據模式自動選擇 sumo 或 sumo-gui
    sumo_binary = "sumo" if is_train_mode else "sumo-gui"
    sumoCmd = [
        sumo_binary,
        "-c", SUMO_CONFIG_FILE,
        "--time-to-teleport", "300",
        "--tripinfo-output", "tripinfo_RL_{}.xml" ,
        "--seed", str(sim_seed), # 【新增】加入隨機種子碼
        
        # 【新增：啟用子車道模型】
        "--lateral-resolution", "0.05" # 設置橫向解析度 (例如：每 0.2m 一個子車道)
    ]
    traci.start(sumoCmd)
    
    
    # --- 3. 初始化並開始模擬 ---
    step = 0
    cumulative_reward = 0.0
    logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
    num_phases = len(logics[0].phases) if logics else 4
    # 【關鍵修正 1】：新增時間追蹤變數
    time_since_last_change = 0
    # 【修正】: 動態獲取 state_size 並建立模型
    lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
    real_state_size = len(list(set(lanes))) + 2
    agent.state_size = real_state_size
    agent.build_models() # 在獲取真實維度後，才建立模型

    print(f"成功獲取交通號誌 '{TRAFFIC_LIGHT_ID}' 的相位總數: {num_phases}")
    print(f"狀態維度 (State Size): {agent.state_size}")

    # --- 4. 主模擬與訓練/測試迴圈 (最終穩定結構) ---
    while step < MAX_SIMULATION_STEPS:
        try:
            # 1. 檢查退出條件
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("所有車輛已離開模擬，提前結束。")
                break
                
            # 1. 推進單步模擬與時間計數 (每一步都執行)
            traci.simulationStep()
            step += 1
            time_since_last_change += 1 

            # 2. 決策與學習邏輯：只在固定的 DECISION_INTERVAL 發生
            if step % DECISION_INTERVAL == 0:
                
                # 2.1 獲取狀態 (在決策點獲取)
                current_state = get_state(TRAFFIC_LIGHT_ID)
                current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                
                action = 0 # 預設：維持
                
                # 2.2 判斷是否允許 RL 決策 (必須是綠燈，且超過最小綠燈時間)
                if current_phase % 2 == 0: # 確保在綠燈相位 (0, 2, ...)
                    
                    if time_since_last_change >= MIN_GREEN_TIME:
                        action = agent.choose_action(current_state)
                    
                    # 2.3 執行切換動作
                    if action == 1:
                        # 切換到下一個相位 (黃燈)
                        next_phase = (current_phase + 1) % num_phases
                        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
                        time_since_last_change = 0 # 重置計時器
                
                # 2.4 學習與紀錄 (發生在每個決策點)
                next_state = get_state(TRAFFIC_LIGHT_ID)
                reward, current_total_queue_length = calculate_reward(TRAFFIC_LIGHT_ID)
                
                if is_train_mode:
                    agent.learn(current_state, action, reward, next_state) 
                
                cumulative_reward += reward
                
                # 2.5 輸出紀錄 (只在決策點輸出)
                time_info = f" | Phase Time: {time_since_last_change:.1f}s"
                phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
        
                if is_train_mode:
                    status_line = f"時間: {step}s{time_info} | 獎勵: {reward:.2f} | States: {phase_state} | Action: {action} | Epsilon: {agent.exploration_rate:.3f}"
                else:
                    status_line = f"時間: {step}s{time_info} | 瞬間獎勵: {reward:.2f}  | States: {phase_state} | 排隊總數: {current_total_queue_length:.2f}"
                
                print(status_line, flush=True)

            # 3. 處理非 RL 控制的相位跳轉 (黃燈 -> 紅燈/綠燈)
            #    (此處不需要任何額外的程式碼，由 SUMO 內部處理)

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
        print(f"最終累積獎勵: {cumulative_reward:.2f}")
    notification.notify(
        title = "Python RL Trainning Finish",
        message = f"RUN PID: {os.getpid()}, MODEL ID= {instance_id}" ,
            
        # displaying time
        timeout=100 # seconds
    )


# --- 程式進入點 ---
if __name__ == "__main__":
    get_sumo_home()
    # run_experiment()
    # --- 6. 命令行參數處理：新增一個參數來控制模式 ---
    main()
    print("程式執行完畢！")