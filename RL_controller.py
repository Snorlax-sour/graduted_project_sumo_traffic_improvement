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
    
# context: 在程式的全局區域

def calculate_reward(tls_id, last_arrived_count):
    """計算獎勵：主要獎勵通行量，輕微懲罰等待時間"""

    # 1. 獎勵：這一輪新到達終點的車輛數 (直接對應吞吐量)
    current_arrived_count = traci.simulation.getArrivedNumber()
    reward_throughput = (current_arrived_count - last_arrived_count) * 10.0 # 給予更高的權重

    # 2. 懲罰：當前路口的總等待時間 (作為一個輔助目標)
    lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
    waiting_penalty = -total_waiting_time * 0.1 # 使用一個較小的權重
    # --- (新增部分：安全性懲罰) ---
    # 3. 碰撞懲罰：偵測是否有車禍發生
    colliding_vehicles = traci.simulation.getCollidingVehiclesNumber()
    if colliding_vehicles > 0:
        # 如果發生了碰撞，給予一個巨大的負分懲罰
        safety_penalty = -100.0
    else:
        # 如果沒有發生碰撞，不給懲罰 (給予少量正分可能會讓AI學會什麼都不做)
        safety_penalty = 0.0
    # --- (新增部分結束) ---
    total_reward = reward_throughput + waiting_penalty + safety_penalty
    
    return total_reward, current_arrived_count
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
    # context: 在 run_experiment 函式的內部
    
    # --- 4. 主模擬與學習迴圈 ---
    step = 0
    last_arrived_vehicles = 0 # 追踪上一轮的到达车辆数

    while step < 10000:
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("所有車輛已離開模擬，提前結束。")
                break

            current_state = get_state(TRAFFIC_LIGHT_ID)
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)

            action = 0
            if current_phase % 2 == 0 and step > MIN_GREEN_TIME: # 稍微调整判断逻辑
                action = agent.choose_action(current_state)

            if action == 1 and current_phase % 2 == 0:
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
                
                # 等待黃燈時間
                for _ in range(3):
                    traci.simulationStep()
                    step += 1
            else:
                # (修正点): 如果决定维持，就让模拟继续跑 DECISION_INTERVAL 步
                for _ in range(DECISION_INTERVAL):
                # 在进入下一步之前，先检查模拟是否已经没有车辆
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break # 如果没有车了，就跳出这个小循环
                traci.simulationStep()
                step += 1
            
            # --- 學習步驟 ---
            next_state = get_state(TRAFFIC_LIGHT_ID)
            
            # 计算奖励并更新到达车辆数
            reward, last_arrived_vehicles = calculate_reward(TRAFFIC_LIGHT_ID, last_arrived_vehicles)
            
            agent.learn(current_state, action, reward, next_state)

            if step % 10 == 0:
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