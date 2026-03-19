import traci
import sys
import json # 👑 新增：用於儲存訓練紀錄
import os
from DQN_RL_Agent import DQNAgent 
import csv 
from plyer import notification 
from datetime import datetime 
#  👇 新增這行：匯入共用工具模組
import sumo_utils 

GA_RESULT_PATH = "./GA_best_result.csv"
# last_total_waiting_time = 0.0 utils replace
# last_total_queue_length = 0.0
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

def get_state(tls_id):
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    unique_lanes = list(set(lanes))
    
    # 1. 靜止排隊車輛 (AI 知道哪裡在塞車)
    halting_counts = [traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes]
    # 2. 👑 新增：車道上的總車輛數 (AI 才會看到正在衝過來的車流！)
    total_vehicles = [traci.lane.getLastStepVehicleNumber(lane) for lane in unique_lanes]
    
    current_phase = traci.trafficlight.getPhase(tls_id)
    GA_min_time_suggestion = GA_OPTIMAL_PHASES[0] if current_phase == 0 else GA_OPTIMAL_PHASES[1]
    
    # 把這兩組數據合併交給 AI
    state_list = halting_counts + total_vehicles + [current_phase, GA_min_time_suggestion]
    return tuple(state_list)

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

# ==============================================================================
# 👑 【核心重構】：獨立出單局執行的函數，讓外面可以跑迴圈
# ==============================================================================
def run_single_episode(episode_num, agent, sumoCmd, is_train_mode, instance_id, mode_label, TRAFFIC_LIGHT_ID):
    print(f"\n" + "-"*55)
    print(f"🏁 正在啟動第 {episode_num} 局模擬 (Episode {episode_num}) ...")
    print("-"*55)
    
    # 1. 啟動 SUMO 模擬器
    traci.start(sumoCmd)
    
    # 2. 🚨 致命關鍵：每局開始前務必重置全域變數，避免分數被上一局干擾！
    sumo_utils.reset_global_state()
    
    # 3. 初始化單局的變數
    
    MAX_SIMULATION_STEPS = 8000
    MIN_GREEN_TIME = 10 
    ACTION_INTERVAL = 10 
    STOP_THRESHOLD = 10  
    
    step = 0
    cumulative_reward = 0.0
    active_crashes = {}
    step_collision_counter = 0 
    time_in_current_phase = 0    
    target_phase_duration = 0    
    last_state = None            
    last_action = None           
    control_mode = "RL"               
    continuous_reward_drop = 0        
    ga_override_cycles_left = 0       
    last_phase = -1                   
    empty_step_counter = 0

    # 取得路口資訊並確認模型是否已建立
    logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
    num_phases = len(logics[0].phases) if logics else 4
    current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
    
    # 動態調整 State Size (如果還沒建立過 Model)
    if not hasattr(agent, "is_built") or not agent.is_built:
        lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
        agent.state_size = len(list(set(lanes))) * 2 + 2# 👑 乘以 2 因為有 queue_lengths 和 total_vehicles
        agent.build_models()
        agent.is_built = True

    # ==========================================
    # 🏃 進入時間步進迴圈 (單局開始)
    # ==========================================
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
            collisions = sumo_utils.detect_real_collisions(TRAFFIC_LIGHT_ID, active_crashes, step)
            step_collision_counter += collisions
            sumo_utils.update_crash_vehicles(active_crashes)
            deadlock_penalty, _ = sumo_utils.handle_deadlock_vehicles(deadlock_threshold=90)
                        
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
                            # 🚨 修正：已改回正確的 calculate_reward
                            reward, _ = sumo_utils.calculate_reward(TRAFFIC_LIGHT_ID, step_collision_counter, time_in_current_phase)
                            step_collision_counter = 0 # 👑 結算後歸零
                            reward += deadlock_penalty
                            cumulative_reward += reward
                            phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                            
                            if sumo_utils.check_downstream_jam(TRAFFIC_LIGHT_ID, 0.85):
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
                                else:
                                    # 👑 只要拿到一次正獎勵，立刻恢復完整權限
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
                        reward, _ = sumo_utils.calculate_reward(TRAFFIC_LIGHT_ID, step_collision_counter, time_in_current_phase)
                        cumulative_reward += reward
                        phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                        
                        # 👑 讓 GA 也面對真實的成績單！
                        if reward < 0:
                            continuous_reward_drop += 1
                        else:
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
            print("SUMO 連線中斷，提前結束單局迴圈。")
            break
            
    print(f"正在關閉第 {episode_num} 局模擬...")
    traci.close()
    
    # 4. 單局結束後的存檔動作
    if is_train_mode:
        agent.save_model() 
        new_count = get_and_update_training_stats(instance_id, increment=True)
        print(f"✅ 第 {episode_num} 局訓練完成並存檔！(累積總經驗次數: {new_count})")
        print(f"📊 本局最終累積獎勵: {cumulative_reward:.2f}")
    else:
        print(f"✅ 第 {episode_num} 局測試完成！")
        print(f"📊 模擬總步數: {step}")
        print(f"📊 本局最終累積獎勵: {cumulative_reward:.2f}")
        
    return cumulative_reward

# ==============================================================================
# 🚀 主程式 (負責準備環境與控制迴圈)
# ==============================================================================
def main():
    sumo_utils.get_sumo_home()
    mode, instance_id = parse_arguments()
    is_train_mode = (mode == 'train')
    
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
    if is_train_mode:
        print(f"🎓 歷史訓練次數: 已經歷過 {current_train_count} 局經驗") 
    print(f"📝 日誌檔案: {log_filename}")
    print("#"*55 + "\n")
    
  # =========================================================================
    # 👑 【核心重構】：動態探勘路口狀態維度 (打破 Hardcode 寫死的 state_size)
    # =========================================================================
    TRAFFIC_LIGHT_ID = "1253678773"
    SUMO_CONFIG_FILE = "osm.sumocfg"
    ACTION_SPACE = [0, 1]
    
    print("🕵️‍♂️ 正在派遣偵察兵探勘 SUMO 實際路口特徵...")
    temp_sumo_cmd = sumo_utils.build_sumo_cmd(
        config_file=SUMO_CONFIG_FILE,
        use_gui=False,       # 隱形模式，不開 GUI
        tripinfo_file=None,  # 探勘不需要產出 XML
        seed=42,
        quiet=True           # 安靜模式，不噴多餘的 Log
    )
    
    # 短暫啟動 SUMO 來獲取路口資訊
    traci.start(temp_sumo_cmd)
    lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
    unique_lanes_count = len(list(set(lanes)))
    traci.close() # 探勘完畢，立刻撤退關閉
    
    # 🧠 計算真正的 State Size
    # 公式：(每條車道的靜止車數) + (每條車道的總車數) + 1(目前相角) + 1(GA建議)
    DYNAMIC_STATE_SIZE = (unique_lanes_count * 2) + 2
    
    print(f"✅ 探勘完成！偵測到路口共有 {unique_lanes_count} 條獨立車道。")
    print(f"🧠 動態設定 Keras 神經網路 State Size: {DYNAMIC_STATE_SIZE}")

    # 👑 現在可以安全、完美地建立 Agent 了！它會自動適應任何地圖！
    print(f"使用的 RL 實例 ID (instance_id): {instance_id}")
    agent = DQNAgent(state_size=DYNAMIC_STATE_SIZE, action_space=ACTION_SPACE, instance_id=instance_id) 
    # ========================================================================= 

    # 👑 2. 載入模型權重
    if is_train_mode:
        print("💡 模式：DQN 訓練模式 (Train Mode)。")
        read_ga_optimal_phases(GA_RESULT_PATH)
        if agent.load_model():
            print("✅ 找到上次訓練模型，將繼續訓練。")
            print(f"🔥 [參數繼承] 讀取 Epsilon: {agent.exploration_rate:.4f}，衰減率 {agent.exploration_decay}")
        else:
            print("⚠️ 未找到模型檔案，將從頭開始訓練。")
    else: 
        print("💡 模式：DQN 測試模式 (Test Mode)。")
        if not agent.load_model():
            print(f"\n❌ 警告：測試模式下未能找到已訓練的模型檔案。")
            sys.exit(1) 
        agent.exploration_rate = 0.0 
        print(f"✅ 模型載入成功。探索率 Epsilon 鎖定為 0。")
        
    # 👑 3. 準備 SUMO 指令
    sim_seed = 42 if is_train_mode else 100 
    sumoCmd = sumo_utils.build_sumo_cmd(
        config_file=SUMO_CONFIG_FILE,
        use_gui=(not is_train_mode), 
        tripinfo_file=f"tripinfo_RL_{instance_id}.xml",
        seed=sim_seed,
        time_to_teleport="3600",
        quiet=False 
    )
    
    mode_label = "[TRAIN]" if is_train_mode else "[TEST]"

    # ==========================================
    # 👑 4. 執行迴圈：精神時光屋啟動！
    # ==========================================
    if is_train_mode:
        TOTAL_EPISODES = 30  # 🎯 這裡可以自由調整你想連續訓練幾局
        print(f"🔥 [啟動精神時光屋] 準備連續訓練 {TOTAL_EPISODES} 局！")
        
        for episode in range(1, TOTAL_EPISODES + 1):
            run_single_episode(episode, agent, sumoCmd, is_train_mode, instance_id, mode_label, TRAFFIC_LIGHT_ID)
            
        print(f"\n🎉 精神時光屋 {TOTAL_EPISODES} 局訓練全數完成！")
        
    else:
        # 測試模式只跑 1 局
        run_single_episode(1, agent, sumoCmd, is_train_mode, instance_id, mode_label, TRAFFIC_LIGHT_ID)
        
    # 👑 5. 全部結束後，發送電腦通知
    try:
        notify_title = "Python RL Trainning Finish" if is_train_mode else "Python RL TEST Finish"
        notification.notify(
            title = notify_title,
            message = f"RUN PID: {os.getpid()}, MODEL ID= {instance_id}" ,
            timeout=10 
        )
    except: pass

if __name__ == "__main__":
    main()
    print("程式執行完畢！")