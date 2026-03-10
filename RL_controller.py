import traci
import sys
import os
from DQN_RL_Agent import DQNAgent 
import csv 
from plyer import notification 
from datetime import datetime  

GA_RESULT_PATH = "./GA_best_result.csv"
last_total_waiting_time = 0.0
last_total_queue_length = 0.0
last_total_cumulative_waiting_time = 0.0

def read_ga_optimal_phases(csv_filepath):
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
        penalty_waiting = current_total_waiting_time * 0.1 
        penalty_queue = 0.5 * (current_total_queue_length ** 2)
        delta_delay = current_total_waiting_time - last_total_waiting_time
        penalty_delta = max(0, delta_delay) * 2.0
        reward = -(penalty_waiting + penalty_queue + penalty_delta)
        last_total_waiting_time = current_total_waiting_time
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
    try:
        upstream_lanes = set(traci.trafficlight.getControlledLanes(tls_id))
        links = traci.trafficlight.getControlledLinks(tls_id)
        actual_downstream_lanes = set()
        for signal_group in links:
            for conn in signal_group:
                down_lane = conn[1] 
                if down_lane not in upstream_lanes:
                    actual_downstream_lanes.add(down_lane)
        
        for lane in actual_downstream_lanes:
            if traci.lane.getLastStepOccupancy(lane) > jam_threshold:
                return True 
        return False
    except Exception as e:
        print(f"下游壅塞偵測錯誤: {e}")
        return False

def print_usage():
    print("\n" + "="*55)
    print("🚦 混合控制 (RL + GA) 交通號誌系統 啟動參數說明")
    print("="*55)
    print("使用方式: python3 RL_controller.py [mode] [instance_id]")
    print("")
    print("參數說明:")
    print("  [mode]        : 執行模式 (train 或 test)")
    print("  [instance_id] : (選填) 模型實例 ID。")
    print("="*55 + "\n")

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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"execute_RL_{timestamp}_{instance_id}_{mode}.txt"

    # 👑 Logger 機制
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    sys.stdout = Logger(log_filename)

    print_usage() 

    print("\n" + "#"*55)
    print(f"🚀 啟動模式: {mode.upper()}")
    print(f"⏰ 開始時間: {timestamp}")
    print(f"🆔 實例 ID : {instance_id}")
    print(f"📝 日誌檔案: {log_filename}")
    print("#"*55 + "\n")
    
    active_crashes = {}
    TRAFFIC_LIGHT_ID = "1253678773"
    SUMO_CONFIG_FILE = "osm.sumocfg"
    MAX_SIMULATION_STEPS = 8000
    MIN_GREEN_TIME = 10 
    ACTION_INTERVAL = 10 
    
    ACTION_SPACE = [0, 1]

    print(f"使用的 RL 實例 ID (instance_id): {instance_id}")
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
        "--collision.mingap-factor", "0", 
        "--collision.action", "none",
    ]
    traci.start(sumoCmd)
    
    step = 0
    cumulative_reward = 0.0
    logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
    num_phases = len(logics[0].phases) if logics else 4
    
    lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
    agent.state_size = len(list(set(lanes))) + 2
    agent.build_models() 

    time_in_current_phase = 0    
    target_phase_duration = 0    
    last_state = None            
    last_action = None           

    control_mode = "RL"               
    continuous_reward_drop = 0        
    ga_override_cycles_left = 0       
    last_phase = -1                   
    mode_label = "[TRAIN]" if is_train_mode else "[TEST]"
    empty_step_counter = 0
    STOP_THRESHOLD = 10  
    
    current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
    
    while step < MAX_SIMULATION_STEPS:
        try:
            traci.simulationStep()
            step += 1
            
            # --- 提早結束與車禍裁判邏輯 (保持不變) ---
            if traci.simulation.getMinExpectedNumber() <= 0:
                empty_step_counter += 1
                if empty_step_counter >= STOP_THRESHOLD:
                    print(f"\n🏁 所有車輛已離開，模擬於第 {step} 秒提早結束。", flush=True)
                    break
            else:
                empty_step_counter = 0  
                
            collisions = traci.simulation.getCollisions()
            for coll in collisions:
                v1, v2 = coll.collider, coll.victim
                if v1 not in active_crashes and v2 not in active_crashes:
                    try:
                        angle_diff = abs(traci.vehicle.getAngle(v1) - traci.vehicle.getAngle(v2)) % 360
                        if angle_diff > 180: angle_diff = 360 - angle_diff
                        if angle_diff > 45 or coll.lane.startswith(':'):
                            active_crashes[v1] = 60 
                            active_crashes[v2] = 60
                    except traci.TraCIException: pass 

            for v in list(active_crashes.keys()):
                active_crashes[v] -= 1
                try:
                    if active_crashes[v] <= 0:
                        traci.vehicle.setSpeed(v, -1) 
                        del active_crashes[v]
                    else:
                        traci.vehicle.setSpeed(v, 0) 
                except traci.TraCIException:
                    del active_crashes[v]

            deadlock_penalty = 0
            for v_id in traci.vehicle.getIDList():
                if traci.vehicle.getSpeed(v_id) < 0.1 and traci.vehicle.getWaitingTime(v_id) > 90:
                    traci.vehicle.remove(v_id) 
                    deadlock_penalty -= 500

            # ==========================================
            # 🚦 核心控制與狀態更新邏輯
            # ==========================================
            new_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            
            # 偵測到燈號發生了實質的切換 (無論是誰切的)
            if new_phase != current_phase:
                current_phase = new_phase
                time_in_current_phase = 0 # 重置計時器為 0
                
                # 奪取 SUMO 控制權 (如果是綠燈，給它一萬秒的壽命，直到 RL 說切換)
                if current_phase % 2 == 0:
                    traci.trafficlight.setPhaseDuration(TRAFFIC_LIGHT_ID, 10000)
            else:
                time_in_current_phase += 1 # 燈號沒變，計時器增加

            # --- 週期切換檢查 (用於 GA 保護機制的倒數) ---
            if current_phase == 0 and last_phase != 0 and last_phase != -1:
                if control_mode == "GA":
                    ga_override_cycles_left -= 1
                    print(f"🔄 GA 接管中... 剩餘 {ga_override_cycles_left} 個週期", flush=True)
                    if ga_override_cycles_left <= 0:
                        print("✅ GA 示範結束，控制權交還給 RL！", flush=True)
                        control_mode = "RL"
                        continuous_reward_drop = 0 
            last_phase = current_phase

            # ==========================================
            # 🟢 綠燈決策階段
            # ==========================================
            if current_phase % 2 == 0: 
                
                if control_mode == "RL":
                    # 只有在 time_in_current_phase > 0 且整除 10 的時候才結算與決策
                    if time_in_current_phase > 0 and time_in_current_phase % ACTION_INTERVAL == 0:
                        current_state = get_state(TRAFFIC_LIGHT_ID)

                        if last_state is not None:
                            reward, _ = calculate_reward(TRAFFIC_LIGHT_ID)
                            cumulative_reward += reward

                            display_drop = continuous_reward_drop 
                            phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                            
                            # 評估是否需要切換到 GA 模式
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

                            # 📢 終於！穩穩地印出 RL 資訊！
                            print(f"{mode_label} 🤖 [RL] 時間: {step}s | 綠燈: {time_in_current_phase}s | {ACTION_INTERVAL}秒獎勵: {reward:.2f} | 掉分: {display_drop}/20 | Epsilon: {agent.exploration_rate:.3f} | 狀態: '{phase_state}'", flush=True)
                            
                            if is_train_mode:
                                agent.learn(last_state, 0, reward, current_state) 

                        # RL 進行下一步決策
                        if time_in_current_phase < MIN_GREEN_TIME:
                            action = 0 
                        else:
                            action = agent.choose_action(current_state)

                        last_state = current_state

                        # RL 決定切換！
                        if action == 1:
                            print(f"⚡ [RL] 決定切換紅綠燈！進入黃燈過渡期。", flush=True)
                            if is_train_mode:
                                agent.learn(current_state, 1, 0, current_state) 
                            
                            # 直接告訴 SUMO 切換下一個相位
                            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, (current_phase + 1) % num_phases)
                            
                # 🟣 GA 控制模式
                elif control_mode == "GA":
                    ga_dur = GA_OPTIMAL_PHASES[0] if current_phase == 0 else GA_OPTIMAL_PHASES[1]
                    
                    if time_in_current_phase > 0 and time_in_current_phase % ACTION_INTERVAL == 0:
                        current_state = get_state(TRAFFIC_LIGHT_ID)
                        reward, _ = calculate_reward(TRAFFIC_LIGHT_ID)
                        cumulative_reward += reward
                        phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                        
                        print(f"{mode_label} 🧬 [GA] 時間: {step}s | 綠燈: {time_in_current_phase}s / {ga_dur}s | {ACTION_INTERVAL}秒獎勵: {reward:.2f} | 掉分: {continuous_reward_drop}/20 | Epsilon: {agent.exploration_rate:.3f} | 狀態: '{phase_state}'", flush=True)

                        if is_train_mode and last_state is not None:
                            agent.learn(last_state, 0, reward, current_state)
                        last_state = current_state

                    # GA 時間到了，強制切換
                    if time_in_current_phase >= ga_dur:
                        print(f"⚡ [GA] 達到最佳秒數 {ga_dur}s，切換紅綠燈！", flush=True)
                        if is_train_mode and last_state is not None:
                            current_state = get_state(TRAFFIC_LIGHT_ID)
                            agent.learn(last_state, 1, 0, current_state)
                            
                        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, (current_phase + 1) % num_phases)

            # ==========================================
            # 🟡🔴 黃燈與紅燈過渡階段
            # ==========================================
            else:
                # 取得 SUMO 預先設定好的黃燈/紅燈秒數 (通常是 3 秒)
                target_phase_duration = traci.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID)
                
                # 黃/紅燈時間到了，我們手動幫它切換到下一個綠燈相位
                if time_in_current_phase >= target_phase_duration:
                    print(f"⏳ [過渡結束] 黃/紅燈 {target_phase_duration}s 結束，切換下一相位。", flush=True)
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, (current_phase + 1) % num_phases)
                    
                    # 關鍵：切換後我們不再等待下一個迴圈，直接強制更新狀態，確保不會卡住
                    current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                    time_in_current_phase = 0
                    
                    # 既然變回綠燈了，立刻鎖死計時器給 RL 控制
                    if current_phase % 2 == 0:
                        traci.trafficlight.setPhaseDuration(TRAFFIC_LIGHT_ID, 10000)

        except traci.TraCIException:
            print("SUMO 連線中斷，提前結束迴圈。")
            break
            
    print("正在關閉模擬...")
    traci.close()
    
    if is_train_mode:
        agent.save_model() 
        try:
            notification.notify(
                title = "Python RL Trainning Finish",
                message = f"RUN PID: {os.getpid()}, MODEL ID= {instance_id}" ,
                timeout=10 
            )
        except: pass
    else:
        print(f"\n✅ 測試完成！使用的模型 ID: {instance_id}")
        print(f"模擬總步數: {step}")
        print(f"最終累積獎勵: {cumulative_reward:.2f}")
        try:
            notification.notify(
                title = "Python RL TEST Finish",
                message = f"RUN PID: {os.getpid()}, MODEL ID= {instance_id}" ,
                timeout=10 
            )
        except: pass

if __name__ == "__main__":
    get_sumo_home()
    main()
    print("程式執行完畢！")