import traci
import sys
import os
from DQN_RL_Agent import DQNAgent # 直接 import class

def get_sumo_home():
    """找到 SUMO 的安装目录并設定環境"""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        return True
    else:
        sys.exit("請確認 SUMO_HOME 環境變數已設定！")

def get_state(tls_id):
    """獲取指定交通號誌的狀態"""
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    unique_lanes = list(set(lanes))
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes]
    return tuple(queue_lengths)
    
def calculate_reward(tls_id):
    """計算指定交通號誌的獎勵 (负等待时间)"""
    lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
    return -total_waiting_time

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
        state_size = len(list(set(lanes)))
        print(f"狀態維度 (State Size): {state_size}")
        
    except traci.TraCIException as e:
        print(f"啟動SUMO或獲取資訊時出錯: {e}")
        traci.close()
        return

    # --- 3. 初始化 Agent ---
    agent = DQNAgent(state_size=state_size, action_space=ACTION_SPACE)

    # --- 4. 主模擬與學習迴圈 ---
    step = 0
    while step < 10000: # 為了快速測試，先跑500步
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("所有車輛已離開模擬，提前結束。")
                break
                
            current_state = get_state(TRAFFIC_LIGHT_ID)
            wait_time_before_action = calculate_reward(TRAFFIC_LIGHT_ID) # 使用-waiting time
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            
            action = 0
            if current_phase % 2 == 0:
                action = agent.choose_action(current_state)
            
            # --- 核心：根據動作，執行並等待固定時間 ---
            if action == 1 and current_phase % 2 == 0:
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
                
                # 等待黃燈時間
                for _ in range(3): # 假設黃燈3秒
                    traci.simulationStep()
                    step += 1
            else:
                # 維持綠燈，等待決策間隔
                for _ in range(DECISION_INTERVAL):
                    traci.simulationStep()
                    step += 1
            
            # --- 學習步驟 ---
            next_state = get_state(TRAFFIC_LIGHT_ID)
            # 使用「差异化」奖励
            wait_time_after_action = calculate_reward(TRAFFIC_LIGHT_ID)
            reward = wait_time_after_action - wait_time_before_action
            
            agent.learn(current_state, action, reward, next_state)

            if step % 10 < 5: # 每10步打印一次資訊，避免洗版
                # 告訴 Python：「執行完這句 print 之後，不要再把內容放進購物車裡等了，立刻、馬上把它送到超市出口（寫入檔案）！」
                print(f"時間: {step}s | 獎勵: {reward:.2f} | Epsilon: {agent.exploration_rate:.3f}", flush=True)

        except traci.TraCIException:
            print("SUMO 連線中斷，提前結束迴圈。")
            break

    # --- 5. 結束模擬 ---
    print("正在關閉模擬...")
    traci.close()

# --- 程式進入點 ---
if __name__ == "__main__":
    get_sumo_home()
    run_experiment()
    print("程式執行完畢！")