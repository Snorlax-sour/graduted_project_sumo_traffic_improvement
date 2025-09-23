import traci
import sys
import os
from RL_Agent import RLAgent
import random

"""
這是整合了所有修正的最終版本。
它包含：
1. 穩定的單次啟動與關閉流程。
2. 引入'最小綠燈時間'來穩定Agent的決策。
3. 使用'差異化獎勵'來提供更有效的學習信號。
"""

# --- 設定 SUMO 環境 ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# --- 模擬設定 ---
SUMO_BINARY = "sumo-gui"
SUMOCFG_PATH = "./osm.sumocfg"
SUMO_CMD = [SUMO_BINARY, "-c", SUMOCFG_PATH, "--quit-on-end", "--time-to-teleport", "-1"]

# --- 全局輔助函式 ---
def get_state(tls_id):
    """獲取指定交通號誌的狀態（各車道排隊車輛數）"""
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    unique_lanes = list(set(lanes))
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes]
    return tuple(queue_lengths)

def get_total_waiting_time(tls_id):
    """獲取指定路口所有車道的總等待時間"""
    lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
    return total_waiting_time

# --- 主程式 ---
def run_simulation():
    """主模擬迴圈"""

    TRAFFIC_LIGHT_ID = "1253678773"
    ACTION_SPACE = [0, 1]  # 0: 維持, 1: 切換
    MIN_GREEN_TIME = 10    # 綠燈最少持續時間

    try:
        # 1. 啟動 SUMO，這是整個程式唯一一次啟動
        traci.start(SUMO_CMD)
        
        # --- 在連線建立後，獲取號誌資訊並初始化 Agent ---
        logics = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
        if logics:
            num_phases = len(logics[0].phases)
            print(f"成功獲取交通號誌 '{TRAFFIC_LIGHT_ID}' 的相位總數: {num_phases}")
        else:
            print(f"錯誤：找不到ID為'{TRAFFIC_LIGHT_ID}'的號誌邏輯，使用預設值4。")
            num_phases = 4 

        agent = RLAgent(action_space=ACTION_SPACE)

        # --- 2. 模擬主迴圈 ---
        step = 0
        phase_timer = 0 # 用來計時當前相位已持續多久
        
        while step < 3000:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("所有車輛已離開模擬，提前結束。")
                break

            # 記錄「行動前」的狀態和等待時間
            current_state = get_state(TRAFFIC_LIGHT_ID)
            wait_time_before_action = get_total_waiting_time(TRAFFIC_LIGHT_ID)
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            
            # 只有在綠燈相位(偶數位)且已達最小綠燈時間，才讓 Agent 決策
            if current_phase % 2 == 0 and phase_timer >= MIN_GREEN_TIME:
                action = agent.choose_action(current_state)
                
                if action == 1:
                    # 切換到下一個相位並重置計時器
                    next_phase = (current_phase + 1) % num_phases
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
                    phase_timer = 0
            else:
                action = 0 # 維持現狀
            
            # --- 讓模擬時間前進一步 ---
            traci.simulationStep()
            step += 1
            phase_timer += 1
            # --- 時間前進完畢 ---

            # 獲取「行動後」的結果並計算獎勵
            wait_time_after_action = get_total_waiting_time(TRAFFIC_LIGHT_ID)
            reward = wait_time_before_action - wait_time_after_action
            
            next_state = get_state(TRAFFIC_LIGHT_ID)
            agent.learn(current_state, action, reward, next_state)

            if step % 10 == 0:
                print(f"時間: {step}s | 獎勵: {reward:.2f} | Epsilon: {agent.exploration_rate:.3f}")

    except traci.TraCIException as e:
        print(f"模擬中斷，錯誤: {e}")

    finally:
        # --- 3. 結束模擬 ---
        # 確保連線一定會被關閉
        print("正在關閉模擬...")
        traci.close()
        sys.stdout.flush()


# --- 程式進入點 ---
if __name__ == "__main__":
    run_simulation()
    print("模擬結束！")