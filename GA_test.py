import traci
from traci._trafficlight import Logic, Phase
import sys
import os
import csv
from datetime import datetime

# --- 基礎設定 ---
ACTION_INTERVAL = 10
TRAFFIC_LIGHT_ID = "1253678773"
SUMO_CONFIG_FILE = "osm.sumocfg"
MAX_SIMULATION_STEPS = 8000
SIM_SEED = 100  # 測試基準種子 (與 Baseline 保持一致以確保公平)
GA_RESULT_PATH = "./GA_best_result.csv"

# 用於計算 Reward 的全域變數
last_total_waiting_time = 0.0
last_total_queue_length = 0.0

def get_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        return True
    else:
        sys.exit("請確認 SUMO_HOME 環境變數已設定！")

def read_ga_optimal_phases(csv_filepath):
    """從 GA 輸出的 CSV 檔案中讀取最後一行的最佳秒數"""
    DEFAULT_PHASES = [35.0, 25.0] 
    if not os.path.exists(csv_filepath):
        print(f"⚠️ 警告：找不到 GA 結果檔案 '{csv_filepath}'，使用預設 {DEFAULT_PHASES}。")
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
            print(f"✅ 成功讀取 GA 最佳解：[綠燈一: {phase1}s, 綠燈二: {phase2}s]")
            return [phase1, phase2]
    except Exception as e:
        print(f"讀取 GA 結果錯誤 ({e})，使用預設 {DEFAULT_PHASES}。")
        return DEFAULT_PHASES

# ==========================================
# 📊 獎勵與壅塞計算函數
# ==========================================
def get_total_queue_length(tls_id):
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        total_queue_length = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes])
        return total_queue_length
    except Exception:
        return 0

def calculate_reward(tls_id, collision_count=0):
    global last_total_waiting_time
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        current_total_waiting_time = sum(
            [traci.vehicle.getWaitingTime(veh_id) for lane in unique_lanes for veh_id in traci.lane.getLastStepVehicleIDs(lane)]
        )
        current_total_queue_length = get_total_queue_length(tls_id)
        penalty_waiting = current_total_waiting_time * 0.1 
        penalty_queue = 0.5 * (current_total_queue_length ** 2)
        delta_delay = current_total_waiting_time - last_total_waiting_time
        penalty_delta = max(0, delta_delay) * 2.0
        # 👑 統一懲罰標準
        penalty_collision = collision_count * 1000.0  # 一次車禍扣 1000 分
        
        reward = -(penalty_waiting + penalty_queue + penalty_delta + penalty_collision)
        last_total_waiting_time = current_total_waiting_time
        return reward, current_total_queue_length
    except Exception:
        return 0.0, 0.0

def check_downstream_jam(tls_id, jam_threshold=0.85):
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
    except Exception:
        return False

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    get_sumo_home()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"execute_GATEST_{timestamp}.txt"
    
    # 將 print 導向檔案與終端機
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

    print("\n" + "="*55)
    print("🚦 啟動 GA 最佳化測試 (靜態注入 GA 最佳秒數 + 智能裁判)")
    print(f"⏰ 開始時間: {timestamp}")
    print(f"📝 日誌檔案: {log_filename}")
    print("="*55 + "\n")

    ga_phases = read_ga_optimal_phases(GA_RESULT_PATH)

    sumoCmd = [
        "sumo", "-c", SUMO_CONFIG_FILE,
        "--time-to-teleport", "300",
        "--tripinfo-output", f"tripinfo_GATEST_{timestamp}.xml",
        "--seed", str(SIM_SEED),
        "--lateral-resolution", "0.05",
        "--collision.mingap-factor", "0",
        "--collision.action", "warn",
        "--no-warnings", "true",
        "--no-step-log", "true",
        "--collision.check-junctions", "true", # 加強路口判定
    ]
    
    traci.start(sumoCmd)

    # 👑 【核心】：奪取並覆寫 SUMO 的紅綠燈配置
    try:
        all_logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
        if all_logics:
            default_logic = all_logics[0]
            orig_phase0_state = default_logic.phases[0].state 
            orig_phase2_state = ""
            for p in default_logic.phases:
                if 'G' in p.state and p.state != orig_phase0_state:
                    orig_phase2_state = p.state
                    break
            if not orig_phase2_state:
                orig_phase2_state = orig_phase0_state.replace('G', 'r').replace('g', 'r').replace('r', 'G')

            logic = Logic(
                programID="ga_prog",
                phases=[            
                    Phase(ga_phases[0], orig_phase0_state),
                    Phase(3, default_logic.phases[1].state if len(default_logic.phases) > 1 else orig_phase0_state.replace('G', 'y')),
                    Phase(ga_phases[1], orig_phase2_state), 
                    Phase(3, default_logic.phases[3].state if len(default_logic.phases) > 3 else orig_phase2_state.replace('G', 'y'))
                ],
                type=0,
                currentPhaseIndex=0
            )
            traci.trafficlight.setProgramLogic(TRAFFIC_LIGHT_ID, logic)
            traci.trafficlight.setProgram(TRAFFIC_LIGHT_ID, logic.programID)
            
            print(f"🚥 [GA 覆寫資訊] 已成功將紅綠燈配置改為：")
            print(f"   👉 相位 0 (🟢): {ga_phases[0]:4.1f} 秒 | '{orig_phase0_state}'")
            print(f"   👉 相位 2 (🟢): {ga_phases[1]:4.1f} 秒 | '{orig_phase2_state}'")
    except Exception as e:
        print(f"覆寫 GA 紅綠燈失敗: {e}")
        
    print("-" * 55 + "\n")
    
    step = 0
    active_crashes = {}
    report_true_collisions = []
    report_jam_events = []
    is_currently_jammed = False
    jam_start_time = 0
    empty_step_counter = 0
    STOP_THRESHOLD = 10
    cumulative_reward = 0.0

    current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
    time_in_current_phase = 0

    step_collision_counter = 0
    # 👑 在 main 裡面先取得受控路口的所有車道清單
    controlled_lanes = set(traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID))
    step_collision_counter = 0 # 用於每 10 秒結算一次
    while step < MAX_SIMULATION_STEPS:
        try:
            traci.simulationStep()
            step += 1
            
            new_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            if new_phase != current_phase:
                current_phase = new_phase
                time_in_current_phase = 0 
            else:
                time_in_current_phase += 1 

            if traci.simulation.getMinExpectedNumber() <= 0:
                empty_step_counter += 1
                if empty_step_counter >= STOP_THRESHOLD:
                    print(f"\n🏁 所有車輛已離開，模擬於第 {step} 秒提早結束。")
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
                except: del active_crashes[v]

            # 救災防死鎖與計算罰款
            deadlock_penalty = 0
            for v_id in traci.vehicle.getIDList():
                if traci.vehicle.getSpeed(v_id) < 0.1:
                    if traci.vehicle.getWaitingTime(v_id) > 90:
                        traci.vehicle.remove(v_id)
                        deadlock_penalty -= 500

            # 偽裝成 RL 輸出給畫圖腳本
            if step % ACTION_INTERVAL == 0:
                reward, _ = calculate_reward(TRAFFIC_LIGHT_ID, step_collision_counter)
                step_collision_counter = 0 # 歸零
                reward += deadlock_penalty 
                cumulative_reward += reward
                phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                
                jammed = check_downstream_jam(TRAFFIC_LIGHT_ID, jam_threshold=0.85)
                if jammed and not is_currently_jammed:
                    is_currently_jammed = True
                    jam_start_time = step
                elif not jammed and is_currently_jammed:
                    is_currently_jammed = False
                    report_jam_events.append((jam_start_time, step))
                # 👑 【新增】：真實計算 GA 的連續負獎勵掉分次數
                if reward < 0:
                    continuous_reward_drop += 1
                else:
                    continuous_reward_drop = 0
                # 👑 偽裝輸出 (包含 time_in_current_phase)
                print(f"[GA_TEST]  [GA] 時間: {step}s | 綠燈: {time_in_current_phase}s | {ACTION_INTERVAL}秒獎勵: {reward:.2f} | 掉分: {continuous_reward_drop}/20 | Epsilon: 0.000 | 狀態: '{phase_state}'", flush=True)

        except traci.TraCIException:
            print("SUMO 連線中斷。")
            break

    traci.close()
    if is_currently_jammed: report_jam_events.append((jam_start_time, step))

    # ==========================================
    # 📊 總結報告
    # ==========================================
    print("\n" + "="*55)
    print("📊 GA 最佳化測試 專屬分析報告")
    print("="*55)
    print(f"⏱️ 1. 模擬總耗時: {step} 秒")
    print(f"💰    總累積獎勵: {cumulative_reward:.2f} 分 (比較對象: Baseline)")
    
    print(f"\n🛑 2. 下游壅塞發生記錄 (共 {len(report_jam_events)} 次):")
    for idx, (start, end) in enumerate(report_jam_events, 1):
        print(f"   - 區間 {idx}: 第 {start}s ~ {end}s (持續 {end - start}s)")

    print(f"\n💥 3. 真車禍發生記錄 (共 {len(report_true_collisions)} 次):")
    for idx, (t, v1, v2, lane) in enumerate(report_true_collisions, 1):
        print(f"   - 第 {t}s | {v1} 撞擊 {v2} | 地點: {lane}")
    print("="*55)
    print(f"📄 測試資料與相容 Log 已存檔: {log_filename}")

if __name__ == "__main__":
    main()