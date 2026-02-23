import traci
import sys
import os
from DQN_RL_Agent import DQNAgent 
import csv 
from plyer import notification 

GA_RESULT_PATH = "./GA_best_result.csv"
last_total_waiting_time = 0.0
last_total_queue_length = 0.0
last_total_cumulative_waiting_time = 0.0

def read_ga_optimal_phases(csv_filepath):
    """從 GA 輸出的 CSV 檔案中讀取最後一行的 phase1 和 phase2 數值"""
    DEFAULT_PHASES = [35.0, 25.0] 
    if not os.path.exists(csv_filepath):
        print(f"警告：找不到 GA 結果檔案 '{csv_filepath}'，使用預設 {DEFAULT_PHASES}。")
        return DEFAULT_PHASES
    try:
        with open(csv_filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) < 2: 
                return DEFAULT_PHASES
            header = rows[0]
            last_row = rows[-1]
            phase1 = float(last_row[header.index('phase1')])
            phase2 = float(last_row[header.index('phase2')])
            print(f"✅ 成功讀取 GA 最佳解：[{phase1}, {phase2}]")
            return [phase1, phase2]
    except Exception as e:
        print(f"讀取 GA 結果錯誤 ({e})，使用預設 {DEFAULT_PHASES}。")
        return DEFAULT_PHASES
    
GA_OPTIMAL_PHASES = read_ga_optimal_phases(GA_RESULT_PATH)

def get_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        return True
    else:
        sys.exit("請確認 SUMO_HOME 環境變數已設定！")

def get_state(tls_id):
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    unique_lanes = list(set(lanes))
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes]
    current_phase = traci.trafficlight.getPhase(tls_id)
    GA_min_time_suggestion = 0.0
    
    # 假設 Phase 0 是主幹道，Phase 2 是次幹道
    if current_phase == 0:  
        GA_min_time_suggestion = GA_OPTIMAL_PHASES[0]
    elif current_phase == 2: 
        GA_min_time_suggestion = GA_OPTIMAL_PHASES[1]
    
    state_list = queue_lengths + [current_phase] + [GA_min_time_suggestion]
    return tuple(state_list)

def calculate_reward(tls_id):
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        current_total_waiting_time = 0.0
        for lane in unique_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in vehicle_ids:
                current_total_waiting_time += traci.vehicle.getWaitingTime(veh_id)
        
        global last_total_waiting_time
        delta_delay = last_total_waiting_time - current_total_waiting_time
        delta_delay *= -1
        current_total_queue_length  = get_total_queue_length(tls_id)
        last_total_waiting_time = current_total_waiting_time
        beta = 0.2 
        reward =  delta_delay - (beta * (current_total_queue_length ** 2))
        return reward, current_total_waiting_time
            
    except traci.TraCIException:
        return 0.0, 0.0 
    except AttributeError as e:
        return calculate_reward_queue_fallback(tls_id)
    except Exception as e_general:
        return 0.0, 0.0

def get_total_queue_length(tls_id):
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        total_queue_length = 0
        for lane in unique_lanes:
            total_queue_length += traci.lane.getLastStepHaltingNumber(lane)
        return total_queue_length
    except Exception:
        return 0

def calculate_reward_queue_fallback(tls_id):
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        current_total_queue_length = 0.0
        for lane in unique_lanes:
            current_total_queue_length += traci.lane.getLastStepHaltingNumber(lane)
        
        global last_total_waiting_time
        delta_queue = last_total_waiting_time - current_total_queue_length
        last_total_waiting_time = current_total_queue_length
        reward = delta_queue * 1.0 
        return reward, current_total_queue_length
    except Exception:
        return 0.0, 0.0

def parse_arguments():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ['train', 'test']:
            print("❌ 錯誤: 第一個參數必須是 'train' 或 'test'。")
            sys.exit(1)
        instance_id = sys.argv[2] if len(sys.argv) > 2 else "default_id"
    else: 
        mode = 'train'
        instance_id = "default_id"
    return mode, instance_id

def main():
    mode, instance_id = parse_arguments()
    is_train_mode = (mode == 'train')

    # 參數設定
    TRAFFIC_LIGHT_ID = "1253678773"
    SUMO_CONFIG_FILE = "osm.sumocfg"
    MAX_SIMULATION_STEPS = 8000
    MIN_GREEN_TIME = 10 
    
    # 【重構 1：定義新的 Action Space 與偏移量映射】
    # 動作 0 -> -5 秒
    # 動作 1 ->  0 秒 (完全採用 GA 解)
    # 動作 2 -> +5 秒
    OFFSET_MAPPING = {0: -5, 1: 0, 2: 5} 
    ACTION_SPACE = [0, 1, 2] # 動作維度變為 3
    
    print(f"使用的 RL 實例 ID (instance_id): {instance_id}")
    # 注意：你的 DQNAgent 需要支援 action_size=3
    agent = DQNAgent(state_size=6, action_space=ACTION_SPACE, instance_id=instance_id) 

    if is_train_mode:
        print("💡 模式：DQN 訓練模式 (Train Mode)。")
        read_ga_optimal_phases(GA_RESULT_PATH)
        if agent.load_model():
            print("✅ 找到上次訓練模型，將繼續訓練。")
        else:
            print("⚠️ 未找到模型檔案，將從頭開始訓練。")
    else: 
        print("💡 模式：DQN 測試模式 (Test Mode)。")
        if not agent.load_model():
            print(f"\n❌ 警告：測試模式下未能找到已訓練的模型檔案。")
            sys.exit(1) 
        agent.exploration_rate = 0.0 
        print(f"✅ 模型載入成功。探索率 Epsilon 鎖定為 0。")
        
    if not get_sumo_home():
        sys.exit(1)
        
    sim_seed = 42 if is_train_mode else 100 
    sumo_binary = "sumo" if is_train_mode else "sumo-gui"
    sumoCmd = [
        sumo_binary, "-c", SUMO_CONFIG_FILE,
        "--time-to-teleport", "300",
        "--tripinfo-output", f"tripinfo_RL_{instance_id}.xml",
        "--seed", str(sim_seed),
        "--lateral-resolution", "0.05" 
    ]
    traci.start(sumoCmd)
    
    step = 0
    cumulative_reward = 0.0
    logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
    num_phases = len(logics[0].phases) if logics else 4
    
    lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
    agent.state_size = len(list(set(lanes))) + 2
    agent.build_models() 

    # 【重構 2：狀態機變數】
    time_in_current_phase = 0    # 在當前相位已經運行的時間
    target_phase_duration = 0    # 當前相位「應該」運行多久
    last_state = None            # 紀錄綠燈剛開始時的狀態
    last_action = None           # 紀錄綠燈剛開始時選擇的動作

    while step < MAX_SIMULATION_STEPS:
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("所有車輛已離開模擬，提前結束。")
                break
                
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            
            # 【重構 3：在「進入新相位的第一秒」做決策】
            if time_in_current_phase == 0:
                if current_phase % 2 == 0: # 如果進入的是綠燈相位 (0 或 2)
                    
                    # 1. 獲取狀態並決策
                    current_state = get_state(TRAFFIC_LIGHT_ID)
                    action = agent.choose_action(current_state)
                    offset = OFFSET_MAPPING[action]
                    
                    # 2. 獲取 GA 基礎時間並計算最終時長
                    ga_base = GA_OPTIMAL_PHASES[0] if current_phase == 0 else GA_OPTIMAL_PHASES[1]
                    target_phase_duration = max(MIN_GREEN_TIME, ga_base + offset)
                    
                    # 3. 儲存 State 與 Action，等待相位結束後學習
                    last_state = current_state
                    last_action = action
                    
                    # (可選) 強制覆蓋 SUMO 內部的該相位時長設定
                    traci.trafficlight.setPhaseDuration(TRAFFIC_LIGHT_ID, target_phase_duration)
                    
                else: 
                    # 如果進入的是黃燈或紅燈相位，直接獲取 SUMO 預設時長 (通常是 3s 或 4s)
                    target_phase_duration = traci.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID)

            # 步進模擬
            traci.simulationStep()
            step += 1
            time_in_current_phase += 1 

            # 【重構 4：檢查當前相位是否該結束了】
            if time_in_current_phase >= target_phase_duration:
                
                # 如果剛剛結束的是「綠燈」，表示 RL 的一個動作執行完畢，可以計算 Reward 並學習
                if current_phase % 2 == 0 and last_state is not None:
                    next_state = get_state(TRAFFIC_LIGHT_ID)
                    reward, current_total_queue_length = calculate_reward(TRAFFIC_LIGHT_ID)
                    
                    if is_train_mode:
                        agent.learn(last_state, last_action, reward, next_state) 
                    
                    cumulative_reward += reward
                    
                    # 印出日誌 (每個綠燈結束時印出)
                    phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                    actual_duration = time_in_current_phase
                    if is_train_mode:
                        print(f"時間: {step}s | 綠燈結束 (時長: {actual_duration}s) | 狀態: '{phase_state}' | 獎勵: {reward:.2f} | 動作(偏移): {OFFSET_MAPPING[last_action]}s | Epsilon: {agent.exploration_rate:.3f}", flush=True)
                    else:
                        print(f"時間: {step}s | 綠燈結束 (時長: {actual_duration}s) | 狀態: '{phase_state}' | 獎勵: {reward:.2f} | 動作(偏移): {OFFSET_MAPPING[last_action]}s | 排隊: {current_total_queue_length:.2f}", flush=True)
                # 強制切換到下一個相位 (綠燈 -> 黃燈，或黃燈 -> 綠燈)
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
                
                # 重置計時器，準備迎接新相位
                time_in_current_phase = 0 

        except traci.TraCIException:
            print("SUMO 連線中斷，提前結束迴圈。")
            break
            
    print("正在關閉模擬...")
    traci.close()
    
    if is_train_mode:
        agent.save_model() 
    else:
        print(f"\n✅ 測試完成！使用的模型 ID: {instance_id}")
        print(f"模擬總步數: {step}")
        print(f"最終累積獎勵: {cumulative_reward:.2f}")
        
    notification.notify(
        title = "Python RL Trainning Finish",
        message = f"RUN PID: {os.getpid()}, MODEL ID= {instance_id}" ,
        timeout=100 
    )

if __name__ == "__main__":
    get_sumo_home()
    main()
    print("程式執行完畢！")