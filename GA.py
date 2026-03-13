import traci
from traci._trafficlight import Logic, Phase
from plyer import notification 
import xml.etree.ElementTree as ET
import concurrent.futures 
import os
import sys
from deap import base, creator, tools
import random
import csv
import datetime
import uuid # 【新增】用於產生絕對不重複的檔案名稱

# --- 基礎設定與 SUMO 啟動 ---
def get_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        return True
    else:
        sys.exit("請確認 SUMO_HOME 環境變數已設定！")
get_sumo_home()

TRAFFIC_LIGHT_ID="1253678773"
GA_INSTANCE_ID = f"default_ga_{os.getpid()}"
if len(sys.argv) > 1:
    GA_INSTANCE_ID = sys.argv[1]

sumo_binary = "sumo"
SUMO_CONFIG_FILE="osm.sumocfg"
sim_seed  = 42

def get_total_delay(filename):
    """解析 XML 計算延遲。如果出錯，返回懲罰值。"""
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
    except (FileNotFoundError, ET.ParseError) as e:
        print(f"[{os.getpid()}] 警告：無法解析 XML '{filename}' (錯誤: {e}). 返回極大延遲作為懲罰。", file=sys.stderr)
        return 999999.0 # 給予極大的懲罰值，讓 GA 淘汰這個壞個體
        
    total_waiting_time = 0.0
    for trip in root.findall("tripinfo"):
        if "timeLoss" in trip.attrib:
            timeLoss = float(trip.attrib["timeLoss"])
            total_waiting_time += timeLoss
    return total_waiting_time

def evaluate(individual):
    active_crashes = {} # 記錄真車禍的車輛 ID 與剩餘罰站時間
    pid = os.getpid()
    # 【修復 1】：加入 uuid，確保就算同一個 Worker 處理，檔案名稱也絕對不重複
    run_id = uuid.uuid4().hex[:6] 
    unique_tripinfo = f"tripinfo_{GA_INSTANCE_ID}_PID{pid}_{run_id}.xml"
    connection_label = f"GA_TL_{pid}_{run_id}"

    unique_sumo_cmd = [
        sumo_binary, "-c", SUMO_CONFIG_FILE,
        "--time-to-teleport", "3600",
        "--seed", str(sim_seed),
        "--lateral-resolution", "0.05",
        "--tripinfo-output", unique_tripinfo,
        "--no-warnings", "true", # 減少控制台的噪音
        "--no-step-log", "true",
        "--collision.mingap-factor", "0", # 【新增】放寬碰撞判定，允許極限貼車鑽縫，前後方向
        "--collision.action", "warn", #讓SUMO無視碰撞 (車子的框框互相碰到了) ，而程式處理，是真碰撞還是假碰撞
        "--collision.check-junctions", "true", # 加強路口判定
    ]

    try:
        traci.start(unique_sumo_cmd, label=connection_label) 
        # 使用特定連線，避免多核心打架
        conn = traci.getConnection(connection_label)

        all_logics = conn.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
        if not all_logics:
            conn.close()
            return (999999.0,)
        
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
                Phase(individual[0], orig_phase0_state),
                Phase(3, default_logic.phases[1].state if len(default_logic.phases) > 1 else orig_phase0_state.replace('G', 'y')),
                Phase(individual[1], orig_phase2_state), 
                Phase(3, default_logic.phases[3].state if len(default_logic.phases) > 3 else orig_phase2_state.replace('G', 'y'))
            ],
            type=0,
            currentPhaseIndex=0
        )
        
        conn.trafficlight.setProgramLogic(TRAFFIC_LIGHT_ID, logic)
        conn.trafficlight.setProgram(TRAFFIC_LIGHT_ID, logic.programID)
        total_deadlock_penalty = 0.0 # 👑 新增：紀錄死鎖造成的延遲懲罰
        MAX_SIM_STEPS = 8000
        step = 0
        # 👑 在 main 裡面先取得受控路口的所有車道清單
        controlled_lanes = set(conn.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID))
        total_collision_count = 0  # 👑 新增：這場模擬總共撞了幾次
        while step < MAX_SIM_STEPS and conn.simulation.getMinExpectedNumber() > 0:
            conn.simulationStep()
            step += 1
            # 👇👇👇 智能裁判邏輯 👇👇👇
            # 偵測碰撞
            collisions = conn.simulation.getCollisions()
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
                        angle1, angle2 = conn.vehicle.getAngle(v1), conn.vehicle.getAngle(v2)
                        angle_diff = abs(angle1 - angle2) % 360
                        if angle_diff > 180: angle_diff = 360 - angle_diff
                        
                        if angle_diff > 45 or coll.lane.startswith(':'):
                            print(f"💥 [REAL_COLLISION] 本路口發生車禍! Step: {step} | Lane: {coll.lane}", flush=True)
                            total_collision_count += 1  # 👑 紀錄發生次數
                            active_crashes[v1] = 60
                            active_crashes[v2] = 60
                    except: pass

            # 執行物理路障
            for v in list(active_crashes.keys()):
                active_crashes[v] -= 1
                try:
                    if active_crashes[v] <= 0:
                        conn.vehicle.setSpeed(v, -1) 
                        del active_crashes[v]
                    else:
                        conn.vehicle.setSpeed(v, 0) 
                except traci.TraCIException:
                    del active_crashes[v]
            # 👆👆👆 智能裁判邏輯結束 👆👆👆

            # 👇👇👇 🚑 救災防死鎖與計算罰款 👇👇👇
            # 同樣全部使用 conn. 來操作
            for v_id in conn.vehicle.getIDList():
                if conn.vehicle.getSpeed(v_id) < 0.1:
                    if conn.vehicle.getWaitingTime(v_id) > 90:
                        conn.vehicle.remove(v_id) # 移出死鎖車輛
                        # 💣 因為 GA 是要找「等待時間最少」的，所以懲罰是加上去！
                        # 一台車死鎖，我們就給這個基因組合「增加 500 秒」的等待時間！
                        total_deadlock_penalty += 500.0 
            # 👆👆👆 救災邏輯結束 👆👆👆
        # 【修正 2：加入心跳監視器】
            # 每模擬 500 步就回報一次，讓你知道它沒有死機
            if step % 500 == 0:
                print(f"[PID {pid}] 正在執行模擬... 第 {step}/{MAX_SIM_STEPS} 步 (剩餘車輛: {conn.simulation.getMinExpectedNumber()})", flush=True)

        # 【修復 2】：非常關鍵！必須先關閉連線，SUMO 才會把 XML 寫完！
        conn.close()
        
        # 【修復 3】：確定關閉後，才去讀取 XML
        delay = get_total_delay(unique_tripinfo) 
        # 👑 核心：總延遲 = 原始延遲 + 死鎖罰款 + (車禍次數 * 5000)
        final_penalty_score = delay + total_deadlock_penalty + (total_collision_count * 1000.0)
        
        return (final_penalty_score,)
            
    except Exception as e:
        print(f"Error in PID {pid}: {e}", flush=True)
        return (999999.0,) 
    finally:
        # 【修復 4】：確保連線一定被關閉，並且強制刪除垃圾檔案
        try:
            traci.getConnection(connection_label).close()
        except: 
            pass
        if os.path.exists(unique_tripinfo):
            try:
                os.remove(unique_tripinfo)
            except:
                pass

# --- GA 參數設定與初始化 ---
POP_SIZE = 100 # 建議測試時先調小，確認跑得動再改回 100
GEN_NUM = 50
TIME_MIN = 10
TIME_MAX = 100

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, TIME_MIN, TIME_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate) 
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=TIME_MIN, up=TIME_MAX, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)


# 【修復 5】：主程式保護！這在 Windows 使用多核心是必備的
def main():
    print(f"主程序 PID {os.getpid()}: 啟動 GA 實例 ID: {GA_INSTANCE_ID}", flush=True)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = rf"./GA_{GA_INSTANCE_ID}__{timestamp}.csv"

    csv_file = open(file=filename, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["generation", "phase1", "phase2", "delay", "os_pid"])

    pop = toolbox.population(n=POP_SIZE)
    first_values = 0
    
    # 【修復 A：建立名人堂 (Hall of Fame)】
    # 設定只記憶 1 個歷史表現最好的絕對菁英
    hof = tools.HallOfFame(1) 

    print(f"\n🔁 開始進行 GA 訓練...\n", flush=True)

    # 開啟多核心運算
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(f"\n🔁 開始評估初始群體 (第 0 代)，共 {POP_SIZE} 個體 (多核心加速中...)\n" ,flush=True)
        
        fitnesses = list(executor.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
        # 【修復 B：更新 Generation 0 到名人堂】
        hof.update(pop)
        first_values = hof[0].fitness.values[0] # 記錄初始群體的全局最佳
            
        print(f"✅ Gen 0 初始群體評估完成！\n", flush=True)
        PATIENCE = 10  # 耐性值：如果連續 10 代沒進步就停
        no_improve_count = 0
        best_fitness_so_far = float('inf')
        for gen in range(GEN_NUM):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:   
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print(f"🔄 第 {gen+1} 代：開始評估 {len(invalid_ind)} 個新個體 (多核心加速中...)", flush=True)
            
            new_fitnesses = list(executor.map(toolbox.evaluate, invalid_ind))
            
            for ind, fit in zip(invalid_ind, new_fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            
            # 【修復 C：每一代評估完後，將新群體丟給名人堂檢查】
            # 如果有比歷史最佳更低的 delay，名人堂會自動更新並覆蓋
            hof.update(pop)
            
            # 👑 【新增】提取「當代最佳解」與「全局歷史最佳解」
            current_gen_best = tools.selBest(pop, k=1)[0] # 當代 100 個體中最好的一個
            
            csv_file.flush()
            global_best = hof[0]                          # 歷史以來最好的一個

            print(f"第 {gen+1} 代 當代最佳組合：{current_gen_best}, 等待時間：{current_gen_best.fitness.values[0]:.2f} 秒", flush=True)
            print(f"第 {gen+1} 代 歷史最佳組合：{global_best}, 最小等待時間：{global_best.fitness.values[0]:.2f} 秒", flush=True)

            # 檢查是否有進步
            current_best_fit = hof[0].fitness.values[0]
            csv_writer.writerow([gen + 1, current_gen_best[0], current_gen_best[1], f"{current_gen_best.fitness.values[0]:.2f}", f"{os.getpid()}"])
            csv_file.flush()
            if current_best_fit < best_fitness_so_far:
                best_fitness_so_far = current_best_fit
                no_improve_count = 0  # 有進步，計數重置
                FINAL_RESULT_FILENAME = "./GA_best_result.csv" 
                # 📝 【修改 1】：寫入歷程檔的，永遠是「當代」的最佳組合與延遲
                

                try:
                    # 📝 【修改 2】：寫入 GA_best_result.csv 的，永遠是「全局歷史 (HOF)」的最佳解
                    with open(FINAL_RESULT_FILENAME, mode="w", newline="", encoding="utf-8") as final_f:
                        final_writer = csv.writer(final_f)
                        final_writer.writerow(["generation", "phase1", "phase2", "delay", "os_pid"])
                        final_writer.writerow([gen + 1, global_best[0], global_best[1], f"{global_best.fitness.values[0]:.2f}", f"{os.getpid()}"])
                except Exception as e:
                    print(f"error write best csv file: {e}", flush=True)
            else:
                no_improve_count += 1 # 沒進步，耐性扣點
                
            print(f"第 {gen+1} 代，連續未進步：{no_improve_count}/{PATIENCE}")

            # 觸發提前停止
            if no_improve_count >= PATIENCE:
                print(f" [!] 偵測到演算法已收斂，提前停止於第 {gen+1} 代。")
                break
           

            

    # 輸出結果
    final_global_best = hof[0]
    print("\n✅ 訓練完成！")
    print(f"最佳紅綠燈時間組合為：{final_global_best}")
    print(f"總等待時間：{final_global_best.fitness.values[0]:.2f} 秒")
    print(f"第一代等待時間：{first_values:.2f} 秒")

    try:
        notification.notify(
            title = "Python GA Trainning Finish",
            message = f"RUN PID: {os.getpid()} , MODEL ID= GA {timestamp}" ,
            timeout=10 
        )
    except:
        pass 
        
    csv_file.close()
    print(f"\n📄 已將所有結果寫入 {filename}")

# --- 啟動保護 (必要) ---
if __name__ == '__main__':
    main()