import traci
import sys
import os
# import csv 沒用到
from datetime import datetime
import sumo_utils

# --- 基礎設定 ---
ACTION_INTERVAL = 10
TRAFFIC_LIGHT_ID = "1253678773"
SUMO_CONFIG_FILE = "osm.sumocfg"
MAX_SIMULATION_STEPS = 8000
SIM_SEED = 100  # 測試基準種子







# ==========================================
# 🚀 主程式
# ==========================================
def main():
    sumo_utils.get_sumo_home()
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

    sumoCmd = sumo_utils.build_sumo_cmd(
        config_file=SUMO_CONFIG_FILE,
        use_gui=False,
        tripinfo_file=f"tripinfo_BASELINE_{timestamp}.xml",
        seed=SIM_SEED,
        time_to_teleport="3600",
        quiet=False  # 原本有 --no-warnings 和 --no-step-log
    )
    
    traci.start(sumoCmd)
    # 👑 【精確版】：抓取並印出 SUMO 地圖預設的每一個紅綠燈相位狀態
    sumo_utils.reset_global_state()  # 👈 重要！
    try:
        # 取得地圖上「所有」紅綠燈的 ID，這樣多路口也能用！
        tls_ids = traci.trafficlight.getIDList()
        
        for tls_id in tls_ids:
            sumo_utils.force_static_traffic_light(tls_id)
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
    # 👇 新增這個計數器 👇
    continuous_reward_drop = 0
    # 👑 在 main 裡面先取得受控路口的所有車道清單
    controlled_lanes = set(traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID))
    step_collision_counter = 0 # 用於每 10 秒結算一次
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

           ## 偵測碰撞
            collisions = sumo_utils.detect_real_collisions(TRAFFIC_LIGHT_ID, active_crashes, step)
            step_collision_counter += collisions
            # 3. 救災防死鎖與計算罰款
            sumo_utils.update_crash_vehicles(active_crashes)
            deadlock_penalty = sumo_utils.handle_deadlock_vehicles()
            # 4. 每 5 秒輸出一次偽裝成 RL 的 Log，餵給畫圖腳本！
            if step % ACTION_INTERVAL == 0:
                # 這裡也要修改 calculate_reward 讓它能接收車禍數
                reward, _ = sumo_utils.calculate_reward(TRAFFIC_LIGHT_ID, step_collision_counter, time_in_current_phase)
                step_collision_counter = 0 # 歸零
                reward += deadlock_penalty 
                cumulative_reward += reward
                phase_state = traci.trafficlight.getRedYellowGreenState(TRAFFIC_LIGHT_ID)
                
                # 偵測壅塞以觸發 plot_results 的紫色背景
                jammed = sumo_utils.check_downstream_jam(TRAFFIC_LIGHT_ID, jam_threshold=0.85)
                if jammed and not is_currently_jammed:
                    print("🚨 下游癱瘓，強制切換 GA 疏導！", flush=True) # 觸發畫圖的紫色區塊
                    is_currently_jammed = True
                    jam_start_time = step
                elif not jammed and is_currently_jammed:
                    print("✅ GA 示範結束，控制權交還給 RL！", flush=True) # 結束畫圖的紫色區塊
                    is_currently_jammed = False
                    report_jam_events.append((jam_start_time, step))
                # 👇👇👇 👑 新增：真實計算連續負獎勵掉分次數 👇👇👇
                if reward < 0:
                    continuous_reward_drop += 1
                else:
                    continuous_reward_drop = 0
                # 👑 【核心】：列印出與 RL 完美相容的正規表示式 Log
                # 這樣 plot_results.py 的 `時間:\s*(\d+)s \| 綠燈.*? \| 5秒獎勵:\s*(-?\d+\.\d+) \| 掉分:\s*(\d+)/20 \| Epsilon:\s*(\d+\.\d+)` 就能抓到！
                print(f"[BASELINE] 時間: {step}s | 綠燈: {time_in_current_phase}s | {ACTION_INTERVAL}秒獎勵: {reward:.2f} | 掉分: {continuous_reward_drop}/20 | Epsilon: 0.000 | 狀態: '{phase_state}'", flush=True)

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