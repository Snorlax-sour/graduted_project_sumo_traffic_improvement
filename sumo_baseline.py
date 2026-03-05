import traci
import sys
import os
import csv
from datetime import datetime

# --- 基礎設定 ---
ACTION_INTERVAL = 10
TRAFFIC_LIGHT_ID = "1253678773"
SUMO_CONFIG_FILE = "osm.sumocfg"
MAX_SIMULATION_STEPS = 8000
SIM_SEED = 100  # 測試基準種子

# 用於計算 Reward 的全域變數 (與 RL_controller.py 完全一致)
last_total_waiting_time = 0.0
last_total_queue_length = 0.0

def get_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        return True
    else:
        sys.exit("請確認 SUMO_HOME 環境變數已設定！")

# ==========================================
# 📊 獎勵與壅塞計算函數 (完全拷貝自 RL_controller，確保公平)
# ==========================================
def get_total_queue_length(tls_id):
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        total_queue_length = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes])
        return total_queue_length
    except Exception:
        return 0

def calculate_reward(tls_id):
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

        reward = -(penalty_waiting + penalty_queue + penalty_delta)
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
    log_filename = f"execute_BASELINE_{timestamp}.txt"
    
    # 將 print 導向檔案與終端機，方便畫圖
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
    print("🚦 啟動 BASELINE 基準線測試 (純預設時相 + 智能裁判)")
    print(f"⏰ 開始時間: {timestamp}")
    print(f"📝 日誌檔案: {log_filename}")
    print("="*55 + "\n")

    sumoCmd = [
        "sumo", "-c", SUMO_CONFIG_FILE,
        "--time-to-teleport", "300",
        "--tripinfo-output", f"tripinfo_BASELINE_{timestamp}.xml",
        "--seed", str(SIM_SEED),
        "--lateral-resolution", "0.05",
        "--collision.mingap-factor", "0",
        "--collision.action", "none",
        "--no-warnings", "true",
        "--no-step-log", "true"
    ]
    
    traci.start(sumoCmd)
    # 👑 【精確版】：抓取並印出 SUMO 地圖預設的每一個紅綠燈相位狀態
    try:
        all_logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
        if all_logics:
            default_logic = all_logics[0]
            print(f"🚥 [地圖預設資訊] 讀取到原始紅綠燈配置 (Program ID: '{default_logic.programID}')：")
            
            for i, p in enumerate(default_logic.phases):
                # 判斷這個相位是綠燈階段還是黃/紅燈過渡階段
                if 'G' in p.state or 'g' in p.state:
                    phase_type = "🟢 綠燈階段"
                elif 'y' in p.state or 'Y' in p.state:
                    phase_type = "🟡 黃燈過渡"
                else:
                    phase_type = "🔴 全紅淨空"
                    
                print(f"   👉 相位 {i} ({phase_type}): {p.duration:4.1f} 秒 | 狀態: '{p.state}'")
        else:
            print("🚥 [地圖預設資訊] 找不到任何紅綠燈配置。")
    except Exception as e:
        print(f"無法讀取預設紅綠燈秒數: {e}")
        
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
    # 👑 新增：追蹤目前紅綠燈的相位與持續時間
    current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
    time_in_current_phase = 0
    while step < MAX_SIMULATION_STEPS:
        try:
            traci.simulationStep()
            step += 1
            # 👑 新增：更新相位時間
            new_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            if new_phase != current_phase:
                current_phase = new_phase
                time_in_current_phase = 0 # 燈號切換，秒數歸零
            else:
                time_in_current_phase += 1 # 燈號沒變，秒數 +1
            # 1. 結束判定
            if traci.simulation.getMinExpectedNumber() <= 0:
                empty_step_counter += 1
                if empty_step_counter >= STOP_THRESHOLD:
                    print(f"\n🏁 所有車輛已離開，模擬於第 {step} 秒提早結束。")
                    break
            else:
                empty_step_counter = 0

            # 2. 智能裁判邏輯 (過濾假車禍)
            collisions = traci.simulation.getCollisions()
            for coll in collisions:
                v1, v2 = coll.collider, coll.victim
                if v1 not in active_crashes and v2 not in active_crashes:
                    try:
                        angle1, angle2 = traci.vehicle.getAngle(v1), traci.vehicle.getAngle(v2)
                        angle_diff = abs(angle1 - angle2) % 360
                        if angle_diff > 180: angle_diff = 360 - angle_diff
                        
                        if angle_diff > 45 or coll.lane.startswith(':'):
                            active_crashes[v1] = 60
                            active_crashes[v2] = 60
                            report_true_collisions.append((step, v1, v2, coll.lane))
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

            # 3. 救災防死鎖與計算罰款
            deadlock_penalty = 0
            for v_id in traci.vehicle.getIDList():
                if traci.vehicle.getSpeed(v_id) < 0.1:
                    if traci.vehicle.getWaitingTime(v_id) > 90:
                        traci.vehicle.remove(v_id)
                        deadlock_penalty -= 500

            # 4. 每 5 秒輸出一次偽裝成 RL 的 Log，餵給畫圖腳本！
            if step %  ACTION_INTERVAL == 0:
                reward, _ = calculate_reward(TRAFFIC_LIGHT_ID)
                reward += deadlock_penalty 
                cumulative_reward += reward
                phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                
                # 偵測壅塞以觸發 plot_results 的紫色背景
                jammed = check_downstream_jam(TRAFFIC_LIGHT_ID, jam_threshold=0.85)
                if jammed and not is_currently_jammed:
                    print("🚨 下游癱瘓，強制切換 GA 疏導！", flush=True) # 觸發畫圖的紫色區塊
                    is_currently_jammed = True
                    jam_start_time = step
                elif not jammed and is_currently_jammed:
                    print("✅ GA 示範結束，控制權交還給 RL！", flush=True) # 結束畫圖的紫色區塊
                    is_currently_jammed = False
                    report_jam_events.append((jam_start_time, step))

                # 👑 【核心】：列印出與 RL 完美相容的正規表示式 Log
                # 這樣 plot_results.py 的 `時間:\s*(\d+)s \| 綠燈.*? \| 5秒獎勵:\s*(-?\d+\.\d+) \| 掉分:\s*(\d+)/20 \| Epsilon:\s*(\d+\.\d+)` 就能抓到！
                print(f"[BASELINE] 🤖 [RL] 時間: {step}s | 綠燈: {step}s | 5秒獎勵: {reward:.2f} | 掉分: 0/20 | Epsilon: 0.000 | 狀態: '{phase_state}'", flush=True)

        except traci.TraCIException:
            print("SUMO 連線中斷。")
            break

    traci.close()
    if is_currently_jammed: report_jam_events.append((jam_start_time, step))

    # ==========================================
    # 📊 輸出你要求的獨立客製化報告
    # ==========================================
    print("\n" + "="*55)
    print("📊 BASELINE 基準線專屬分析報告")
    print("="*55)
    print(f"⏱️ 1. 模擬總耗時: {step} 秒")
    print(f"💰    總累積獎勵: {cumulative_reward:.2f} 分 (可與 RL 的最終得分比較)")
    
    print(f"\n🛑 2. 下游壅塞發生記錄 (共 {len(report_jam_events)} 次):")
    for idx, (start, end) in enumerate(report_jam_events, 1):
        print(f"   - 區間 {idx}: 第 {start}s ~ {end}s (持續 {end - start}s)")

    print(f"\n💥 3. 真車禍發生記錄 (共 {len(report_true_collisions)} 次):")
    for idx, (t, v1, v2, lane) in enumerate(report_true_collisions, 1):
        print(f"   - 第 {t}s | {v1} 撞擊 {v2} | 地點: {lane}")
    print("="*55)
    print(f"📄 基準線資料與相容 Log 已存檔: {log_filename}")

if __name__ == "__main__":
    main()