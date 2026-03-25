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
import uuid # 用於產生絕對不重複的檔案名稱
import sumo_utils

# =====================================================================
# 👑 新增：全局 Log 紀錄器 (攔截所有 print 並同時寫入終端機與 TXT)
# =====================================================================
if "GA_START_TIME" not in os.environ:
    os.environ["GA_START_TIME"] = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

LOG_FILENAME = f"execute_GA_{os.environ['GA_START_TIME']}.txt"

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(LOG_FILENAME)
sys.stderr = sys.stdout 
# =====================================================================

sumo_utils.get_sumo_home()

TRAFFIC_LIGHT_ID="1253678773"
GA_INSTANCE_ID = f"default_ga_{os.getpid()}"
if len(sys.argv) > 1:
    GA_INSTANCE_ID = sys.argv[1]

sumo_binary = "sumo"
SUMO_CONFIG_FILE="osm.sumocfg"
sim_seed  = 42


def evaluate(individual):
    active_crashes = {}
    pid = os.getpid()
    run_id = uuid.uuid4().hex[:6] 
    unique_tripinfo = f"tripinfo_{GA_INSTANCE_ID}_PID{pid}_{run_id}.xml"
    connection_label = f"GA_TL_{pid}_{run_id}"

    unique_sumo_cmd = sumo_utils.build_sumo_cmd(
        config_file=SUMO_CONFIG_FILE,
        use_gui=False,
        tripinfo_file=unique_tripinfo,
        seed=sim_seed,
        time_to_teleport="3600",
        quiet=True
    )

    try:
        traci.start(unique_sumo_cmd, label=connection_label) 
        sumo_utils.reset_global_state()  
        conn = traci.getConnection(connection_label)

        all_logics = conn.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)
        if not all_logics:
            conn.close()
            return (999999.0, 0, 0)
        
        default_logic = all_logics[0]
        orig_phase0_state = default_logic.phases[0].state 
        orig_phase2_state = ""
        
        for p in default_logic.phases:
            if 'G' in p.state and p.state != orig_phase0_state:
                orig_phase2_state = p.state
                break
        
        if not orig_phase2_state:
            orig_phase2_state = orig_phase0_state.replace('G', 'r').replace('g', 'r').replace('r', 'G')
            
        orig_yellow_dur1 = default_logic.phases[1].duration if len(default_logic.phases) > 1 else 3.0
        orig_yellow_dur2 = default_logic.phases[3].duration if len(default_logic.phases) > 3 else 3.0
        
        logic = Logic(
            programID="ga_prog",
            phases=[            
                Phase(individual[0], orig_phase0_state),
                Phase(orig_yellow_dur1, default_logic.phases[1].state if len(default_logic.phases) > 1 else orig_phase0_state.replace('G', 'y')),
                Phase(individual[1], orig_phase2_state), 
                Phase(orig_yellow_dur2, default_logic.phases[3].state if len(default_logic.phases) > 3 else orig_phase2_state.replace('G', 'y'))
            ],
            type=0,
            currentPhaseIndex=0
        )
        
        conn.trafficlight.setProgramLogic(TRAFFIC_LIGHT_ID, logic)
        conn.trafficlight.setProgram(TRAFFIC_LIGHT_ID, logic.programID)
        
        total_deadlock_penalty = 0.0
        total_deadlock_count = 0
        total_collision_count = 0
        MAX_SIM_STEPS = 8000
        step = 0
        # --- 【新增：動態懲罰追蹤器初始化】 ---
        junction_retention_dict = {}       # 紀錄車輛在路口內的存留時間 {vid: seconds}
        total_jvr_seconds = 0.0            # 改為記錄總滯留秒數
        total_phase_transition_leftovers = 0 # 變燈殘留車輛數
        total_downstream_jam_seconds = 0.0 # 改為記錄下游壅塞秒數

        last_phase = conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
        # ------------------------------------
        while step < MAX_SIM_STEPS and conn.simulation.getMinExpectedNumber() > 0:
            conn.simulationStep()
            step += 1
            
            # 1. JVR 滯留計時
            active_vids = conn.vehicle.getIDList()
            for vid in list(junction_retention_dict.keys()):
                if vid not in active_vids or not conn.vehicle.getRoadID(vid).startswith(':'):
                    del junction_retention_dict[vid]

            for vid in active_vids:
                if conn.vehicle.getRoadID(vid).startswith(':'):
                    junction_retention_dict[vid] = junction_retention_dict.get(vid, 0) + 1
                    if junction_retention_dict[vid] > 5:
                        total_jvr_seconds += 1.0 # 只計秒數，不次方累加

            # 2. 相位過渡
            current_phase = conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            if current_phase != last_phase:
                in_junction_count = sum(1 for v in junction_retention_dict.keys())
                total_phase_transition_leftovers += in_junction_count
                last_phase = current_phase

            # 3. 下游壅塞
            if sumo_utils.check_downstream_jam(TRAFFIC_LIGHT_ID, jam_threshold=0.85, conn=conn):
                total_downstream_jam_seconds += 1.0 # 只計秒數

            # 碰撞與死鎖 (維持原樣)
            new_collisions = sumo_utils.detect_real_collisions(TRAFFIC_LIGHT_ID, active_crashes, step, conn=conn)
            total_collision_count += new_collisions
            sumo_utils.update_crash_vehicles(active_crashes, conn=conn)
            
            dl_penalty, dl_count = sumo_utils.handle_deadlock_vehicles(deadlock_threshold=90, conn=conn, return_count=True)
            total_deadlock_penalty += dl_penalty  
            total_deadlock_count += dl_count
            
            if step % 500 == 0:
                print(f"[PID {pid}] 正在執行模擬... 第 {step}/{MAX_SIM_STEPS} 步 (剩餘車輛: {conn.simulation.getMinExpectedNumber()})")

        conn.close()
        delay = sumo_utils.calculate_delay(tripinfo_filename=unique_tripinfo)
        
        # ========================================================
        # ⚖️ 【修改：結算制懲罰 (讓總分回到萬級別)】
        # ========================================================
        # 真正的 Delay 約在 5000~20000 之間。我們讓懲罰稍微痛一點，但不要上百萬。
        w_delay = 1.0
        w_deadlock = 50000.0  # 死鎖一次罰 5萬 (非常痛)
        w_collision = 10000.0 # 車禍一次罰 1萬
        w_jvr = 50.0          # 每滯留 1 秒罰 50
        w_transition = 1000.0 # 每次變燈殘留 1 台車罰 1000
        w_downstream = 50.0   # 下游塞車每秒罰 50

        final_penalty_score = (
            (delay * w_delay) + 
            (total_deadlock_count * w_deadlock) + # 改用 count 算，比較乾淨
            (total_collision_count * w_collision) +
            (total_jvr_seconds * w_jvr) + 
            (total_phase_transition_leftovers * w_transition) + 
            (total_downstream_jam_seconds * w_downstream)
        )

        print(f"[PID {pid}] Delay: {delay:.1f} | Col: {total_collision_count} | DL: {total_deadlock_count} | Score: {final_penalty_score:.1f}")
        return (final_penalty_score, total_collision_count, total_deadlock_count, pid) 
            
    except Exception as e:
        print(f"Error in PID {pid}: {e}")
        return (999999.0, 0, 0, pid) 
    finally:
        try: traci.getConnection(connection_label).close()
        except: pass
        if os.path.exists(unique_tripinfo):
            try: os.remove(unique_tripinfo)
            except: pass

# --- GA 參數設定與初始化 ---
POP_SIZE = 30 
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
toolbox.register("mutate", tools.mutUniformInt, low=TIME_MIN, up=TIME_MAX, indpb=0.2)# 將突變率從 0.5 改為 0.2 (0.5 太高了，會破壞菁英基因)
toolbox.register("select", tools.selTournament, tournsize=2)


def main():
    print(f"主程序 PID {os.getpid()}: 啟動 GA 實例 ID: {GA_INSTANCE_ID}")
    print(f"📝 本次訓練日誌將自動寫入: {LOG_FILENAME}")
    
    historical_best_delay = float('inf')
    FINAL_RESULT_FILENAME = "./GA_best_result.csv"
    if os.path.exists(FINAL_RESULT_FILENAME):
        try:
            with open(FINAL_RESULT_FILENAME, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None) 
                row = next(reader, None)
                if row and len(row) >= 4:
                    historical_best_delay = float(row[3])
                    print(f"💾 成功載入歷史最佳紀錄：{historical_best_delay} 分")
        except Exception as e:
            print(f"⚠️ 讀取歷史紀錄失敗: {e}")

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = rf"./GA_{GA_INSTANCE_ID}__{timestamp}.csv"

    csv_file = open(file=filename, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    # 👑 【進階】新增 individual_idx (個體編號) 與 worker_pid (代跑核心)
    csv_writer.writerow(["generation", "individual_idx", "phase1", "phase2", "total_score", "collisions", "deadlocks", "worker_pid"])
    pop = toolbox.population(n=POP_SIZE)
    # 👑 【新增：國王帶兵 (Warm Start)】
    # 把你之前最好的基因塞進第一代，避免沉沒成本浪費
    pop[0] = creator.Individual([92, 20])
    first_values = 0
    hof = tools.HallOfFame(1) 

    print(f"\n🔁 開始進行 GA 訓練...\n")
    safe_workers = max(1, os.cpu_count() - 2)
    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
        print(f"\n🔁 開始評估初始群體 (第 0 代)，共 {POP_SIZE} 個體 (多核心加速中...)\n")
        
        # 👑 【核心修正】：攔截初始群體的多重輸出
        results = list(executor.map(toolbox.evaluate, pop))
        for ind, res in zip(pop, results):
            ind.fitness.values = (res[0],) # 只把總分塞給 GA
            ind.col_cnt = res[1]           # 掛上車禍名牌
            ind.dl_cnt = res[2]            # 掛上死鎖名牌
            ind.worker_pid = res[3]        # 👑 掛上工人名牌 (NEW)


        hof.update(pop)
        first_values = hof[0].fitness.values[0] 
        # 👑 【新增】：將 Gen 0 的 100 個個體全部寫入 CSV
        for idx, ind in enumerate(pop):
            csv_writer.writerow([0, idx, ind[0], ind[1], f"{ind.fitness.values[0]:.2f}", ind.col_cnt, ind.dl_cnt, ind.worker_pid])
        csv_file.flush()
        print(f"✅ Gen 0 初始群體評估完成！\n")
        PATIENCE = 15  
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
            print(f"🔄 第 {gen+1} 代：開始評估 {len(invalid_ind)} 個新個體 (多核心加速中...)")
            
            # 👑 【核心修正】：攔截新世代的多重輸出
            new_results = list(executor.map(toolbox.evaluate, invalid_ind))
            for ind, res in zip(invalid_ind, new_results):
                ind.fitness.values = (res[0],) # 只把總分塞給 GA
                ind.col_cnt = res[1]           # 更新車禍名牌
                ind.dl_cnt = res[2]            # 更新死鎖名牌
                ind.worker_pid = res[3]        # 👑 掛上工人名牌 (NEW)
            pop[:] = offspring
            hof.update(pop)
            
            current_gen_best = tools.selBest(pop, k=1)[0] 
            csv_file.flush()
            global_best = hof[0]                          

            # 👑 印出時加上名牌資訊
            print(f"第 {gen+1} 代 當代最佳：{current_gen_best}, 總分: {current_gen_best.fitness.values[0]:.2f} (車禍:{current_gen_best.col_cnt}, 死鎖:{current_gen_best.dl_cnt})")
            print(f"第 {gen+1} 代 歷史最佳：{global_best}, 總分: {global_best.fitness.values[0]:.2f} (車禍:{global_best.col_cnt}, 死鎖:{global_best.dl_cnt})")

            current_best_fit = hof[0].fitness.values[0]
            # ------------------------------------------------------------
            # 👑 【核心新增】：將本代 100 個個體全部寫入 CSV！
            # ------------------------------------------------------------
            for idx, ind in enumerate(pop):
                # 如果這個個體是菁英保留(沒有重新評估)，他可能帶有舊的 worker_pid，安全讀取
                w_pid = getattr(ind, 'worker_pid', os.getpid())
                csv_writer.writerow([gen + 1, idx, ind[0], ind[1], f"{ind.fitness.values[0]:.2f}", ind.col_cnt, ind.dl_cnt, w_pid])
            csv_file.flush()
            # ------------------------------------------------------------
            
            if current_best_fit < best_fitness_so_far:
                best_fitness_so_far = current_best_fit
                no_improve_count = 0  
                
                if current_best_fit < historical_best_delay:
                    print(f"🎉 突破跨世代歷史紀錄！({historical_best_delay:.2f} 降至 {current_best_fit:.2f})，更新檔案！")
                    historical_best_delay = current_best_fit  
                    try:
                        with open(FINAL_RESULT_FILENAME, mode="w", newline="", encoding="utf-8") as final_f:
                            final_writer = csv.writer(final_f)
                            # 👑 全局最佳 CSV 也同步寫入名牌資訊
                            final_writer.writerow(["generation", "phase1", "phase2", "total_score", "collisions", "deadlocks", "os_pid"])
                            final_writer.writerow([gen + 1, global_best[0], global_best[1], f"{global_best.fitness.values[0]:.2f}", global_best.col_cnt, global_best.dl_cnt, f"{os.getpid()}"])
                    except Exception as e:
                        print(f"error write best csv file: {e}")
                else:
                    print(f"👍 本次訓練有進步 ({current_best_fit:.2f})，但尚未打破歷史紀錄 ({historical_best_delay:.2f})。")
            else:
                no_improve_count += 1 
                
            print(f"第 {gen+1} 代，連續未進步：{no_improve_count}/{PATIENCE}")
            
            if no_improve_count > 0 and no_improve_count % 4 == 0:  
                print(f"⚠️ 偵測到基因庫同質化，保留歷史最強，其餘重新隨機生成！")
                elite = toolbox.clone(hof[0])
                pop = toolbox.population(n=POP_SIZE)
                pop[0] = elite  
                
                # 👑 【核心防呆修復】：必須立刻把這批新生兒派給多核心去評估，否則下一代一開頭的 select 會崩潰！
                reset_invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                print(f"🔄 正在為 {len(reset_invalid_ind)} 個重生個體進行緊急評估 (多核心加速中...)")
                reset_results = list(executor.map(toolbox.evaluate, reset_invalid_ind))
                
                for ind, res in zip(reset_invalid_ind, reset_results):
                    ind.fitness.values = (res[0],)
                    ind.col_cnt = res[1]
                    ind.dl_cnt = res[2]
                    ind.worker_pid = res[3]
                    
                hof.update(pop) # 更新排行榜  
                
            if no_improve_count >= PATIENCE:
                print(f" [!] 偵測到演算法已收斂，提前停止於第 {gen+1} 代。")
                break

    final_global_best = hof[0]
    print("\n✅ 訓練完成！")
    print(f"最佳紅綠燈時間組合為：{final_global_best}")
    print(f"總等待時間(總分)：{final_global_best.fitness.values[0]:.2f} (車禍:{final_global_best.col_cnt}, 死鎖:{final_global_best.dl_cnt})")
    print(f"第一代等待時間：{first_values:.2f}")

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

if __name__ == '__main__':
    main()