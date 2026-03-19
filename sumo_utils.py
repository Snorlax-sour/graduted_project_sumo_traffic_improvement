"""
SUMO 交通模擬共用工具模組
統一管理：獎勵計算、壅塞偵測、死鎖處理、碰撞偵測
"""
import traci
import os
import sys
import xml.etree.ElementTree as ET
# ==========================================
# 📊 全域變數（用於計算獎勵的增量）
# ==========================================
last_total_waiting_time = 0.0
last_total_queue_length = 0.0
COLLISION_PENALTY = 5000

def reset_global_state():
    """重置全域狀態（每次新模擬開始時呼叫）"""
    global last_total_waiting_time, last_total_queue_length
    last_total_waiting_time = 0.0
    last_total_queue_length = 0.0

# ==========================================
# 🚦 獎勵與壅塞計算
# ==========================================
def get_total_queue_length(tls_id):
    """計算路口的總排隊長度"""
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        total_queue_length = sum([
            traci.lane.getLastStepHaltingNumber(lane) 
            for lane in unique_lanes
        ])
        return total_queue_length
    except Exception:
        return 0

def calculate_reward(tls_id, collision_count=0, time_in_current_phase=0):
    """
    計算當前時間步的獎勵
    
    Args:
        tls_id: 交通號誌 ID
        collision_count: 本時間步內發生的車禍次數
        time_in_current_phase: 當前綠燈持續時間（秒）
    
    Returns:
        (reward, current_queue_length): 獎勵值與當前排隊長度
    """
    global last_total_waiting_time
    
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(lanes))
        
        # 計算當前總等待時間
        current_total_waiting_time = sum([
            traci.vehicle.getWaitingTime(veh_id)
            for lane in unique_lanes
            for veh_id in traci.lane.getLastStepVehicleIDs(lane)
        ])
        
        # 計算當前總排隊長度
        current_total_queue_length = get_total_queue_length(tls_id)
        
        # 懲罰項 1: 等待時間
        penalty_waiting = current_total_waiting_time * 0.1
        
        # 懲罰項 2: 排隊長度（平方懲罰，避免大排隊）
        penalty_queue = 0.5 * (current_total_queue_length ** 2)
        
        # 懲罰項 3: 等待時間增量（如果越等越久）
        delta_delay = current_total_waiting_time - last_total_waiting_time
        penalty_delta = max(0, delta_delay) * 2.0
        
        # 懲罰項 4: 車禍（嚴厲懲罰）
        penalty_collision = collision_count * COLLISION_PENALTY
        
        
        
        # 總獎勵（負數 = 懲罰）
        reward = -(penalty_waiting + penalty_queue + penalty_delta + 
                   penalty_collision)
        
        # 更新全域狀態
        last_total_waiting_time = current_total_waiting_time
        
        return reward, current_total_queue_length
        
    except Exception as e:
        print(f"⚠️ 計算獎勵時發生錯誤: {e}")
        return 0.0, 0.0

def calculate_delay(tripinfo_filename):
    """
    [GA 專用] 從 tripinfo.xml 計算整場模擬的真實總延遲。
    精準包含紅燈靜止、煞車減速、以及跟車緩慢移動的延遲 (timeLoss)。
    """
    if not tripinfo_filename or not os.path.exists(tripinfo_filename):
        print(f"⚠️ 找不到 XML 檔案: {tripinfo_filename}")
        return 999999.0

    total_time_loss = 0.0
    try:
        tree = ET.parse(tripinfo_filename)
        root = tree.getroot()
        for trip in root.findall("tripinfo"):
            if "timeLoss" in trip.attrib:
                total_time_loss += float(trip.attrib["timeLoss"])
        return total_time_loss
    except Exception as e:
        print(f"⚠️ 解析 XML 失敗: {e}")
        return 999999.0  # 給予極大懲罰淘汰這個壞基因


def check_downstream_jam(tls_id, jam_threshold=0.85, conn=None):
    """
    檢查下游車道是否壅塞
    
    Args:
        tls_id: 交通號誌 ID
        jam_threshold: 佔用率閾值（0.85 = 85%）
        conn: TraCI 連線物件（多核心 GA 專用，單核心傳 None）
    
    Returns:
        True 如果下游壅塞，False 否則
    """
    connection = conn if conn is not None else traci
    
    try:
        # 取得上游車道（路口進入端）
        upstream_lanes = set(connection.trafficlight.getControlledLanes(tls_id))
        
        # 取得下游車道（路口離開端）
        links = connection.trafficlight.getControlledLinks(tls_id)
        actual_downstream_lanes = set()
        
        for signal_group in links:
            for conn_link in signal_group:
                down_lane = conn_link[1]  # 連接的目標車道
                if down_lane not in upstream_lanes:  # 排除上游車道
                    actual_downstream_lanes.add(down_lane)
        
        # 檢查下游車道的佔用率
        for lane in actual_downstream_lanes:
            if connection.lane.getLastStepOccupancy(lane) > jam_threshold:
                return True
                
        return False
        
    except Exception:
        return False

# ==========================================
# 💥 碰撞偵測
# ==========================================
def detect_real_collisions(tls_id, active_crashes, step, conn=None):
    """
    偵測真實碰撞（過濾平行擦撞）
    
    Args:
        tls_id: 交通號誌 ID
        active_crashes: 當前活躍碰撞字典 {vehicle_id: remaining_freeze_time}
        step: 當前模擬時間步
        conn: TraCI 連線物件（多核心 GA 專用，單核心傳 None）
    
    Returns:
        collision_count: 本時間步新增的真實碰撞數
    """
    # 👇 支援多核心
    connection = conn if conn is not None else traci
    
    collision_count = 0
    
    # 取得受控路口的所有車道
    controlled_lanes = set(connection.trafficlight.getControlledLanes(tls_id))
    
    try:
        collisions = connection.simulation.getCollisions()
        
        for coll in collisions:
            v1, v2 = coll.collider, coll.victim
            
            # 避免重複計數
            if v1 in active_crashes or v2 in active_crashes:
                continue
            
            # 👑 精準過濾：判斷是否屬於本路口
            is_my_junction = False
            
            if coll.lane in controlled_lanes:
                is_my_junction = True
            elif coll.lane.startswith(':'):  # 路口內部車道
                if tls_id in coll.lane:
                    is_my_junction = True
            
            if not is_my_junction:
                continue  # 別處車禍，跳過
            
            # 👑 角度判定：過濾平行擦撞
            try:
                angle1 = connection.vehicle.getAngle(v1)
                angle2 = connection.vehicle.getAngle(v2)
                angle_diff = abs(angle1 - angle2) % 360
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # 角度差 > 45° 或發生在路口內部 = 真車禍
                if angle_diff > 45 or coll.lane.startswith(':'):
                    print(f"💥 [REAL_COLLISION] 本路口發生車禍! Step: {step} | Lane: {coll.lane}", flush=True)
                    collision_count += 1
                    
                    # 凍結車輛 60 秒
                    active_crashes[v1] = 60
                    active_crashes[v2] = 60
                    
            except:
                pass
    
    except Exception as e:
        print(f"⚠️ 碰撞偵測錯誤: {e}")
    
    return collision_count

def update_crash_vehicles(active_crashes, conn=None):
    """
    更新碰撞車輛狀態（倒數計時 + 解除凍結）
    
    Args:
        active_crashes: 當前活躍碰撞字典（會被原地修改）
        conn: TraCI 連線物件（多核心 GA 專用，單核心傳 None）
    """
    connection = conn if conn is not None else traci
    
    for v in list(active_crashes.keys()):
        active_crashes[v] -= 1
        try:
            if active_crashes[v] <= 0:
                # 解除凍結
                connection.vehicle.setSpeed(v, -1)
                del active_crashes[v]
            else:
                # 持續凍結
                connection.vehicle.setSpeed(v, 0)
        except:
            # 車輛可能已被移除
            del active_crashes[v]

# ==========================================
# 🚑 死鎖處理（瞬移版）
# ==========================================
# ==========================================
# 🚑 死鎖處理（瞬移版）- 👑 已修復紅燈誤判
# ==========================================
def handle_deadlock_vehicles(deadlock_threshold=90, conn=None, return_count=False):
    """
    處理死鎖車輛（極簡精準版）
    👑 核心邏輯：只救援「卡在十字路口內部 (Junction)」超過 90 秒的車輛。
    正常車道上的塞車排隊完全不管，讓 AI 自行承擔 Delay 懲罰！
    """
    connection = conn if conn is not None else traci
    deadlock_penalty = 0
    deadlock_count = 0  
    
    for v_id in connection.vehicle.getIDList():
        # 👑 1. 最優先判斷：這台車是否在路口內部 (SUMO 裡路口內的 edge 開頭是 ':')
        current_edge = connection.vehicle.getRoadID(v_id)
        if not current_edge.startswith(':'):
            continue  # 不在路口內？那只是塞車排隊而已，直接跳過，不管等多久都不處罰！
            
        # 👑 2. 如果在路口內部，且車速幾乎停止
        if connection.vehicle.getSpeed(v_id) < 0.1:
            # 👑 3. 且卡在路口中間超過 90 秒 (這就是真正的死鎖！)
            if connection.vehicle.getWaitingTime(v_id) > deadlock_threshold:
                try:
                    route = connection.vehicle.getRoute(v_id)
                    route_index = connection.vehicle.getRouteIndex(v_id)
                    
                    # 嘗試把它往前推到下一個正常的車道上
                    if route_index + 1 < len(route):
                        next_edge = route[route_index + 1]
                        connection.vehicle.moveTo(v_id, f"{next_edge}_0", 10.0)
                        print(f"pid: {os.getpid()} 🚑 [路口死鎖救援] {v_id} 於路口內卡死，瞬移到 {next_edge} 疏通", flush=True)
                        deadlock_penalty -= 500
                        deadlock_count += 1
                    else:
                        # 如果已經沒路可走，只好強制移除
                        connection.vehicle.remove(v_id)
                        print(f"pid: {os.getpid()} 🚑 [路口死鎖移除] {v_id} 於路口內卡死且無退路，強制移除", flush=True)
                        deadlock_penalty -= 10000
                        deadlock_count += 1
                        
                except Exception as e:
                    try:
                        connection.vehicle.remove(v_id)
                        print(f"pid: {os.getpid()} ❌ [瞬移失敗] {v_id} 強制移除 ({e})", flush=True)
                        deadlock_penalty -= 10000
                        deadlock_count += 1
                    except:
                        pass
    
    # 支援雙模式回傳 (GA 需要次數，RL 只需要分數)
    if return_count:
        return deadlock_penalty, deadlock_count
    return deadlock_penalty

# ==========================================
# 🛠️ SUMO 環境設定
# ==========================================
def get_sumo_home():
    """檢查並設定 SUMO_HOME 環境變數"""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        return True
    else:
        sys.exit("請確認 SUMO_HOME 環境變數已設定！")

def force_static_traffic_light(tls_id):
    """
    強制將交通號誌降級為靜態模式（關閉感應式）
    
    Args:
        tls_id: 交通號誌 ID
    """
    try:
        all_logics = traci.trafficlight.getAllProgramLogics(tls_id)
        if not all_logics:
            return
        
        default_logic = all_logics[0]
        
        # 如果已經是靜態，跳過
        if default_logic.type == 0:
            print(f"🚥 [{tls_id}] 已經是靜態 (Static) 模式。")
            return
        
        print(f"⚠️ [{tls_id}] 偵測到動態模式 (Type={default_logic.type})，強制降級為靜態...")
        
        # 建立靜態版本
        from traci._trafficlight import Logic
        static_logic = Logic(
            programID="forced_static",
            type=0,  # 靜態模式
            currentPhaseIndex=0,
            phases=default_logic.phases
        )
        
        traci.trafficlight.setProgramLogic(tls_id, static_logic)
        traci.trafficlight.setProgram(tls_id, static_logic.programID)
        print(f"✅ [{tls_id}] 降級完成！")
        
    except Exception as e:
        print(f"無法降級紅綠燈: {e}")


def build_sumo_cmd(config_file="osm.sumocfg", use_gui=False, tripinfo_file=None, 
                   seed=42, time_to_teleport="3600", quiet=True):
    """
    統一產出 SUMO 啟動指令 (sumoCmd)
    
    Args:
        config_file: SUMO 設定檔名稱
        use_gui: 是否啟動 sumo-gui (True = sumo-gui, False = sumo)
        tripinfo_file: tripinfo 輸出的檔名 (傳入 None 則不輸出)
        seed: 模擬亂數種子
        time_to_teleport: 死鎖車輛瞬移時間閾值
        quiet: 是否隱藏 SUMO 引擎的警告與 Step log
    """
    binary = "sumo-gui" if use_gui else "sumo"
    
    cmd = [
        binary, "-c", config_file,
        "--time-to-teleport", str(time_to_teleport),
        "--seed", str(seed),
        "--lateral-resolution", "0.05",
        "--collision.mingap-factor", "0",
        "--collision.action", "warn",
        "--collision.check-junctions", "true",  # 加強路口判定
    ]
    
    if tripinfo_file:
        cmd.extend(["--tripinfo-output", tripinfo_file])
        # 👑 【核心修復】：強制記錄還卡在路上的塞車受害者！
        cmd.append("--tripinfo-output.write-unfinished")
        
    if quiet:
        cmd.extend(["--no-warnings", "true", "--no-step-log", "true"])
        
    return cmd