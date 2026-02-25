import traci
import sys
import os
from DQN_RL_Agent import DQNAgent 
import csv 
from plyer import notification 
from datetime import datetime  # 務必在最上方 import

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


def check_downstream_jam(tls_id, jam_threshold=0.9):
    """
    【新增】偵測下個路口（下游車道）是否塞車
    如果平均佔有率超過 80% (0.8)，回傳 True
    """
    try:
        links = traci.trafficlight.getControlledLinks(tls_id)
        # 提取所有離開這個路口的下游車道
        outgoing_lanes = set([link[0][1] for link in links if link])
        
        is_jammed = False
        for lane in outgoing_lanes:
            # 取得車道的空間佔有率 (0.0 ~ 1.0)
            occupancy = traci.lane.getLastStepOccupancy(lane)
            if occupancy >= jam_threshold:
                is_jammed = True
                break
        return is_jammed
    except Exception as e:
        print(f"下游壅塞偵測錯誤: {e}")
        return False

def print_usage():
    """印出命令列使用說明"""
    print("\n" + "="*55)
    print("🚦 混合控制 (RL + GA) 交通號誌系統 啟動參數說明")
    print("="*55)
    print("使用方式: python3 RL_controller.py [mode] [instance_id]")
    print("")
    print("參數說明:")
    print("  [mode]        : 執行模式，可選值為:")
    print("                  - train : 啟動訓練模式 (無 GUI，快速運算並更新模型)")
    print("                  - test  : 啟動測試模式 (有 GUI，不更新模型，Epsilon=0)")
    print("  [instance_id] : (選填) 模型實例 ID。用於區分不同的訓練模型檔案。")
    print("                  預設值為 'default_id'。")
    print("")
    print("範例:")
    print("  1. 預設訓練    : python3 RL_controller.py")
    print("  2. 指定 ID 訓練: python3 RL_controller.py train v2_fast")
    print("  3. 指定 ID 測試: python3 RL_controller.py test v2_fast")
    print("="*55 + "\n")

def parse_arguments():
    # 每次啟動都先印出使用說明
    print_usage()
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
    # --- 新增：模式標題列印 ---
    # 【新增】生成 yyyymmdd_hhmm 格式的時間戳記
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
 
    # 建立一個明確的 log 檔名
    log_filename = f"execute_RL_{timestamp}_{instance_id}_{mode}.txt"
    print("\n" + "#"*55)
    print(f"🚀 啟動模式: {mode.upper()}")
    print(f"⏰ 開始時間: {timestamp}")
    print(f"🆔 實例 ID : {instance_id}")
    print(f"📝 日誌檔案: {log_filename}")
    print("#"*55 + "\n")
    # 參數設定
    TRAFFIC_LIGHT_ID = "1253678773"
    SUMO_CONFIG_FILE = "osm.sumocfg"
    MAX_SIMULATION_STEPS = 8000
    MIN_GREEN_TIME = 10 
    ACTION_INTERVAL = 5 # 【重構重點】每 5 秒讓 RL 決策一次
    
    # 【重構 1：定義新的 Action Space 與偏移量映射】
    # 動作 0 -> -5 秒
    # 動作 1 ->  0 秒 (完全採用 GA 解)
    # 動作 2 -> +5 秒
    # OFFSET_MAPPING = {0: -5, 1: 0, 2: 5} 
    # ACTION_SPACE = [0, 1, 2] # 動作維度變為 3
    # 【重構重點】動作空間變為 2 (0: 保持綠燈, 1: 切換相位)
    ACTION_SPACE = [0, 1]

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
        "--lateral-resolution", "0.05" ,
        "--collision.mingap-factor", "0" # 【新增】放寬碰撞判定，允許極限貼車鑽縫
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

    control_mode = "RL"               # 當前主控權："RL" 或 "GA"
    continuous_reward_drop = 0        # 獎勵連續下降計數器
    ga_override_cycles_left = 0       # GA 接管剩餘週期數
    last_phase = -1                   # 用於計算週期切換
    # 這裡建立日誌標籤
    mode_label = "[TRAIN]" if is_train_mode else "[TEST]"
    empty_step_counter = 0
    STOP_THRESHOLD = 10  # 連續 10 秒沒車才結束
    while step < MAX_SIMULATION_STEPS:
        try:
            
            # 1. 執行模擬步進
            traci.simulationStep()
            step += 1
            time_in_current_phase += 1
            # 2. 【核心修正】強健的提早結束判斷
            # getMinExpectedNumber <= 0 代表地圖沒車且未來也沒車要進場
            if traci.simulation.getMinExpectedNumber() <= 0:
                empty_step_counter += 1
                if empty_step_counter >= STOP_THRESHOLD:
                    print(f"\n🏁 所有車輛已離開，模擬於第 {step} 秒提早結束。", flush=True)
                    break
            else:
                empty_step_counter = 0  # 只要有車，重置計數器
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            # 【機制 1】：週期計算 (當相位從最後一個切回 0 時，算作一個完整週期)
            if current_phase == 0 and last_phase != 0 and last_phase != -1:
                if control_mode == "GA":
                    ga_override_cycles_left -= 1
                    print(f"🔄 GA 接管中... 剩餘 {ga_override_cycles_left} 個週期", flush=True)
                    if ga_override_cycles_left <= 0:
                        print("✅ GA 示範結束，控制權交還給 RL！", flush=True)
                        control_mode = "RL"
                        continuous_reward_drop = 0 # 重置計數器
            last_phase = current_phase

            # # 進入新相位的第一秒做決策
            # if time_in_current_phase == 0:
            #     if current_phase % 2 == 0: # 綠燈相位
                    
            #         current_state = get_state(TRAFFIC_LIGHT_ID)
                    
            #         # 【機制 2】：偵測下游是否塞車
            #         if check_downstream_jam(TRAFFIC_LIGHT_ID, jam_threshold=0.85):
            #             if control_mode == "RL":
            #                 print("🚨 警告：下游路口已癱瘓！強制拔除 RL 控制權，切換至 GA 疏導模式！")
            #                 control_mode = "GA"
            #                 ga_override_cycles_left = 3
                    
            #         # 依據主控權決定動作
            #         if control_mode == "RL":
            #             action = agent.choose_action(current_state)
            #             offset = OFFSET_MAPPING[action]
            #         else:
            #             action = 1 # 1 對應 offset 0 (完全遵照 GA)
            #             offset = 0
                    
            #         ga_base = GA_OPTIMAL_PHASES[0] if current_phase == 0 else GA_OPTIMAL_PHASES[1]
            #         target_phase_duration = max(MIN_GREEN_TIME, ga_base + offset)
                    
            #         last_state = current_state
            #         last_action = action
            #         traci.trafficlight.setPhaseDuration(TRAFFIC_LIGHT_ID, target_phase_duration)
                    
            #     else: 
            #         target_phase_duration = traci.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID)

            # # 步進模擬
            # traci.simulationStep()
            # step += 1
            # time_in_current_phase += 1 
            # ==========================================
            # 核心決策邏輯：綠燈階段
            # ==========================================
            if current_phase % 2 == 0: 
                
                if control_mode == "RL":
                    # 每 5 秒，交警 (RL) 出來巡視一次
                    if time_in_current_phase % ACTION_INTERVAL == 0:
                        current_state = get_state(TRAFFIC_LIGHT_ID)

                        # 1. 如果不是第 0 秒，代表剛剛過去的 5 秒我們選擇了「保持綠燈」，現在來結算這 5 秒的獎勵
                        if time_in_current_phase > 0 and last_state is not None:
                            reward, current_total_queue_length = calculate_reward(TRAFFIC_LIGHT_ID)
                            cumulative_reward += reward

                            ## 監控機制：是否塞車或掉分？
                            display_drop = continuous_reward_drop # 新增顯示專用的變數
                            phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                            
                            if check_downstream_jam(TRAFFIC_LIGHT_ID, jam_threshold=0.85):
                                print("🚨 下游癱瘓，強制切換 GA 疏導！", flush=True)
                                control_mode = "GA"
                                ga_override_cycles_left = 3
                            else:
                                if reward < 0:
                                    continuous_reward_drop += 1
                                    display_drop = continuous_reward_drop 
                                    if continuous_reward_drop >= 20:
                                        print(f"📉 RL 連續負獎勵 20 次，觸發 GA 保護機制！", flush=True)
                                        control_mode = "GA"
                                        ga_override_cycles_left = 3
                                        continuous_reward_drop = 0 
                                else:
                                    continuous_reward_drop = 0
                                    display_drop = 0

                            # 【修正 1】：不管是不是訓練模式，都把這 5 秒的結果印出來！
                            # print(f"🤖 [RL] 時間: {step}s | 綠燈已亮: {time_in_current_phase}s | 5秒獎勵: {reward:.2f} | 掉分: {display_drop}/20 | 狀態: '{phase_state}'", flush=True)
                            # 【修正輸出】：加上 mode_label
                            print(f"{mode_label} 🤖 [RL] 時間: {step}s | 綠燈: {time_in_current_phase}s | 5秒獎勵: {reward:.2f} | 掉分: {display_drop}/20 | 狀態: '{phase_state}'", flush=True)
                            # 只有在訓練模式，才把結果送給大腦學習
                            if is_train_mode:
                                agent.learn(last_state, 0, reward, current_state) 

                        # 2. 如果控制權還在 RL 手上，決定「下一個 5 秒」的動作
                        if control_mode == "RL":
                            # 防呆機制：最少要亮綠燈 10 秒
                            if time_in_current_phase < MIN_GREEN_TIME:
                                action = 0 # 強制保持綠燈
                            else:
                                action = agent.choose_action(current_state)

                            last_state = current_state

                            # 如果 RL 決定切換！
                            if action == 1:
                                # 【修正 2】：不管是不是訓練，都印出切換燈號的提示！
                                print(f"⚡ [RL] 決定切換紅綠燈！進入黃燈過渡期。", flush=True)
                                
                                # 【重要修復】：確保 learn 嚴格包在 train_mode 裡面，防止測試模式崩潰！
                                if is_train_mode:
                                    agent.learn(current_state, 1, 0, current_state) 
                                    
                                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, (current_phase + 1) % num_phases)
                                time_in_current_phase = -1 # 重置計時器

            
            # # 當前相位結束
            # if time_in_current_phase >= target_phase_duration:
            #     if current_phase % 2 == 0 and last_state is not None:
            #         next_state = get_state(TRAFFIC_LIGHT_ID)
            #         reward, current_total_queue_length = calculate_reward(TRAFFIC_LIGHT_ID)
                    
            #         # 【機制 3】：監控獎勵是否連續下降
            #         if reward < 0:
            #             continuous_reward_drop += 1
            #             if continuous_reward_drop >= 20 and control_mode == "RL":
            #                 print(f"📉 警告：RL 連續 {continuous_reward_drop} 次獲得負獎勵！觸發 GA 保護機制。")
            #                 control_mode = "GA"
            #                 ga_override_cycles_left = 3
            #                 continuous_reward_drop = 0
            #         else:
            #             continuous_reward_drop = 0 # 只要有一次正獎勵，就歸零
                    
            #         # 只有在 RL 模式下，或是你想讓 RL 從 GA 的行為中學習時才 learn
            #         # 這裡設定為：RL 就算被拔除控制權，依然在旁邊「觀察」並學習 GA 的動作 (Imitation Learning)
            #         if is_train_mode:
            #             agent.learn(last_state, last_action, reward, next_state) 
                    
            #         cumulative_reward += reward
                    
            #         phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
            #         prefix = "🤖 [RL]" if control_mode == "RL" else "🧬 [GA]"
            #         if is_train_mode:
            #             print(f"{prefix} 時間: {step}s | 狀態: '{phase_state}' | 獎勵: {reward:.2f} | 動作偏移: {OFFSET_MAPPING[last_action]}s | 掉分次數: {continuous_reward_drop}/20", flush=True)

            #     next_phase = (current_phase + 1) % num_phases
            #     traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
            #     time_in_current_phase = 0

        except traci.TraCIException:
            print("SUMO 連線中斷，提前結束迴圈。")
            break
            
    print("正在關閉模擬...")
    traci.close()
    
    if is_train_mode:
        agent.save_model() 
        notification.notify(
        title = "Python RL Trainning Finish",
        message = f"RUN PID: {os.getpid()}, MODEL ID= {instance_id}" ,
        timeout=100 
    )
    else:
        print(f"\n✅ 測試完成！使用的模型 ID: {instance_id}")
        print(f"模擬總步數: {step}")
        print(f"最終累積獎勵: {cumulative_reward:.2f}")
        
        notification.notify(
            title = "Python RL TEST & Trainning Finish",
            message = f"RUN PID: {os.getpid()}, MODEL ID= {instance_id}" ,
            timeout=100 
        )

if __name__ == "__main__":
    get_sumo_home()
    main()
    print("程式執行完畢！")