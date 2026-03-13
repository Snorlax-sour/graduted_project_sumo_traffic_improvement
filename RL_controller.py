import traci
import sys
import json # 👑 新增：用於儲存訓練紀錄
import os
from DQN_RL_Agent import DQNAgent 
import csv 
from plyer import notification 
from datetime import datetime  

GA_RESULT_PATH = "./GA_best_result.csv"
last_total_waiting_time = 0.0
last_total_queue_length = 0.0
last_total_cumulative_waiting_time = 0.0
# 👑 【新增函數】：管理訓練次數的讀取與寫入
def get_and_update_training_stats(instance_id, increment=False):
    stats_file = f"stats_{instance_id}.json"
    data = {"training_count": 0, "last_update": ""}
    
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
        except: pass
    
    if increment:
        data["training_count"] += 1
        data["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(stats_file, 'w') as f:
            json.dump(data, f)
            
    return data["training_count"]


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

def calculate_reward(tls_id, collision_count=0):
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        current_total_waiting_time = 0.0
        for lane in unique_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in vehicle_ids:
                current_total_waiting_time += traci.vehicle.getWaitingTime(veh_id)
        # 👑 新增：車禍重罰 (例如一次撞擊扣 1000 分)
        penalty_collision = collision_count * 1000.0
        global last_total_waiting_time
        delta_delay = last_total_waiting_time - current_total_waiting_time
        delta_delay *= -1
        current_total_queue_length  = get_total_queue_length(tls_id)
        penalty_waiting = current_total_waiting_time * 0.1 
        penalty_queue = 0.5 * (current_total_queue_length ** 2)
        delta_delay = current_total_waiting_time - last_total_waiting_time
        penalty_delta = max(0, delta_delay) * 2.0
        reward = -(penalty_waiting + penalty_queue + penalty_delta + penalty_collision)
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
    # 👑 讀取目前的訓練次數
    current_train_count = get_and_update_training_stats(instance_id, increment=False)
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
    print(f"🎓 訓練次數: 第 {current_train_count} 次經驗") # 👑 啟動時印出
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
            # 🚀 加入這行，明確告訴自己已經覆寫成功！
            print(f"🔥 [參數覆寫] 指揮官介入！Epsilon 強制設定為 {agent.exploration_rate}，衰減率 {agent.exploration_decay}")
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
        "--time-to-teleport", "3600",
        "--tripinfo-output", f"tripinfo_RL_{instance_id}.xml",
        "--seed", str(sim_seed),
        "--lateral-resolution", "0.05" ,
        "--collision.mingap-factor", "0", 
        "--collision.action", "warn",
        "--collision.check-junctions", "true", # 加強路口判定
    ]
    traci.start(sumoCmd)
    
    step = 0
    cumulative_reward = 0.0
    logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
    num_phases = len(logics[0].phases) if logics else 4
    # 👑 在 main 裡面先取得受控路口的所有車道清單
    controlled_lanes = set(traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID))
    step_collision_counter = 0 # 用於每 10 秒結算一次
    lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
    agent.state_size = len(list(set(lanes))) + 2
    agent.build_models() 
    step_collision_counter = 0 # 用於每 10 秒結算一次
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
          
            if traci.simulation.getMinExpectedNumber() <= 0:
                empty_step_counter += 1
                if empty_step_counter >= STOP_THRESHOLD:
                    print(f"\n🏁 所有車輛已離開，模擬於第 {step} 秒提早結束。", flush=True)
                    break
            else:
                empty_step_counter = 0  
                
            # 偵測碰撞
            collisions = traci.simulation.getCollisions()
            for coll in collisions:
                v1, v2 = coll.collider, coll.victim
                if v1 not in active_crashes and v2 not in active_crashes:
                    
                    # 👑 【精準過濾】：判斷車禍車道是否屬於本路口
                    is_my_junction = False
                    if coll.lane in controlled_lanes: # 車道在路口進入端
                        is_my_junction = True
                    elif coll.lane.startswith(':'): # 發生在路口內部 (Internal Lane)
                        # 內部車道名稱通常包含路口 ID，例如 :1253678773_0_0
                        if TRAFFIC_LIGHT_ID in coll.lane:
                            is_my_junction = True
                    
                    if not is_my_junction:
                        continue # 👑 別處撞車，與我無關，跳過

                    try:
                        angle1, angle2 = traci.vehicle.getAngle(v1), traci.vehicle.getAngle(v2)
                        angle_diff = abs(angle1 - angle2) % 360
                        if angle_diff > 180: angle_diff = 360 - angle_diff
                        
                        if angle_diff > 45 or coll.lane.startswith(':'):
                            print(f"💥 [REAL_COLLISION] 本路口發生車禍! Step: {step} | Lane: {coll.lane}", flush=True)
                            step_collision_counter += 1 # 👑 累加給 reward 扣分
                            active_crashes[v1] = 60
                            active_crashes[v2] = 60
                    except: pass


               

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
            vehicles = traci.vehicle.getIDList()
            for v_id in vehicles:
                if traci.vehicle.getSpeed(v_id) < 0.1:
                    waiting_time = traci.vehicle.getWaitingTime(v_id)
                    if waiting_time > 90:
                        traci.vehicle.remove(v_id) 
                        deadlock_penalty -= 500
                        
            new_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            if new_phase != current_phase:
                current_phase = new_phase
                time_in_current_phase = 0 
                if current_phase % 2 == 0:
                    traci.trafficlight.setPhaseDuration(TRAFFIC_LIGHT_ID, 10000)
            else:
                time_in_current_phase += 1 

            # ==========================================
            # 🔄 GA 週期結束交接：實作「5次試錯寬限期」
            # ==========================================
            if current_phase == 0 and last_phase != 0 and last_phase != -1:
                if control_mode == "GA":
                    ga_override_cycles_left -= 1
                    print(f"🔄 GA 接管中... 剩餘 {ga_override_cycles_left} 個週期", flush=True)
                    
                    if ga_override_cycles_left <= 0:
                        # 👑 核心改動：給予 RL 5次機會 (20-10 = 10)
                        # 如果原本掉分是 30，我們會把它強行拉回到 10
                        continuous_reward_drop = 10
                        
                        print(f"✅ GA 示範結束，控制權交還。RL 進入「{20-continuous_reward_drop}次限制試用期」(目前掉分: {continuous_reward_drop}/20)", flush=True)
                        control_mode = "RL"
            last_phase = current_phase
            # ==========================================
            # 🟢 綠燈決策階段
            # ==========================================
            if current_phase % 2 == 0: 
                
                # 🤖 RL 控制模式
                if control_mode == "RL":
                    if time_in_current_phase > 0 and time_in_current_phase % ACTION_INTERVAL == 0:
                        current_state = get_state(TRAFFIC_LIGHT_ID)

                        if last_state is not None:
                            # 傳入這段期間發生的車禍總數
                            reward, _ = calculate_reward(TRAFFIC_LIGHT_ID, step_collision_counter)
                            step_collision_counter = 0 # 👑 結算後歸零
                            cumulative_reward += reward
                            phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                            
                            if check_downstream_jam(TRAFFIC_LIGHT_ID, jam_threshold=0.85):
                                print("🚨 下游癱瘓，強制切換 GA 疏導！", flush=True)
                                control_mode = "GA"
                                ga_override_cycles_left = 3
                            else:
                                # 👑 真實計分邏輯
                                if reward < 0:
                                    continuous_reward_drop += 1
                                    
                                    # 如果掉分超過 20，直接強制交給 GA！
                                    if continuous_reward_drop >= 20:
                                        print(f"📉 RL 連續負獎勵 {continuous_reward_drop} 次，觸發 GA 保護機制！", flush=True)
                                        control_mode = "GA"
                                        ga_override_cycles_left = 3
                                        # 🚨 同樣移除這裡的歸零邏輯
                                else:
                                    # 👑 只要拿到一次正獎勵，代表 RL 找到解法了，立刻恢復完整權限
                                    if continuous_reward_drop > 0:
                                        print(f"🌟 RL 表現回升 (Reward > 0)，正式通過試用期，掉分紀錄歸零。", flush=True)
                                    continuous_reward_drop = 0
                                    
                            print(f"{mode_label} 🤖 [RL] 時間: {step}s | 綠燈: {time_in_current_phase}s | {ACTION_INTERVAL}秒獎勵: {reward:.2f} | 掉分: {continuous_reward_drop}/20 | Epsilon: {agent.exploration_rate:.3f} | 狀態: '{phase_state}'", flush=True)
                            
                            if is_train_mode:
                                agent.learn(last_state, 0, reward, current_state) 

                        if control_mode == "RL":
                            if time_in_current_phase < MIN_GREEN_TIME:
                                action = 0 
                            else:
                                action = agent.choose_action(current_state)

                            last_state = current_state

                            if action == 1:
                                print(f"⚡ [RL] 決定切換紅綠燈！進入黃燈過渡期。", flush=True)
                                if is_train_mode:
                                    agent.learn(current_state, 1, 0, current_state) 
                                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, (current_phase + 1) % num_phases)
                                
                # 🧬 GA 控制模式
                elif control_mode == "GA":
                    ga_dur = GA_OPTIMAL_PHASES[0] if current_phase == 0 else GA_OPTIMAL_PHASES[1]
                    
                    if time_in_current_phase > 0 and time_in_current_phase % ACTION_INTERVAL == 0:
                        current_state = get_state(TRAFFIC_LIGHT_ID)
                        reward, _ = calculate_reward(TRAFFIC_LIGHT_ID)
                        cumulative_reward += reward
                        phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                        
                        # 👑 讓 GA 也面對真實的成績單！
                        if reward < 0:
                            continuous_reward_drop += 1
                        else:
                            # 只有 GA 真的排解了壅塞（拿到正獎勵），才算是「真的好了」
                            continuous_reward_drop = 0
                            
                        print(f"{mode_label} 🧬 [GA] 時間: {step}s | 綠燈: {time_in_current_phase}s / {ga_dur}s | {ACTION_INTERVAL}秒獎勵: {reward:.2f} | 掉分: {continuous_reward_drop}/20 | Epsilon: {agent.exploration_rate:.3f} | 狀態: '{phase_state}'", flush=True)

                        if is_train_mode and last_state is not None:
                            agent.learn(last_state, 0, reward, current_state)
                        last_state = current_state

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
                target_phase_duration = traci.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID)
                if time_in_current_phase >= target_phase_duration:
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, (current_phase + 1) % num_phases)
                    
                    current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                    time_in_current_phase = 0
                    
                    if current_phase % 2 == 0:
                        traci.trafficlight.setPhaseDuration(TRAFFIC_LIGHT_ID, 10000)

        except traci.TraCIException:
            print("SUMO 連線中斷，提前結束迴圈。")
            break
            
    print("正在關閉模擬...")
    traci.close()
    
    if is_train_mode:
        agent.save_model() 
        # 👑 訓練完成後，更新紀錄檔中的次數！
        new_count = get_and_update_training_stats(instance_id, increment=True)
        print(f"\n✅ 訓練完成！次數已從 {current_train_count} 更新為 {new_count}。")
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