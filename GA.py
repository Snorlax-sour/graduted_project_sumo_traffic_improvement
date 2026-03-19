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
        
        while step < MAX_SIM_STEPS and conn.simulation.getMinExpectedNumber() > 0:
            conn.simulationStep()
            step += 1
            
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
        # ⚖️ 【完全對齊 RL 物理法則】
        # ========================================================
        # 👑 請確認 sumo_utils 裡的 COLLISION_PENALTY 是 5000 或您設定的數值
        # 這裡為了保險，我們直接手動定義車禍分數 5000 
        RL_COLLISION_PENALTY = 5000.0 
        
        final_penalty_score = delay + abs(total_deadlock_penalty) + (total_collision_count * RL_COLLISION_PENALTY)
        
        return (final_penalty_score, total_collision_count, total_deadlock_count)
            
    except Exception as e:
        print(f"Error in PID {pid}: {e}")
        return (999999.0, 0, 0) 
    finally:
        try: traci.getConnection(connection_label).close()
        except: pass
        if os.path.exists(unique_tripinfo):
            try: os.remove(unique_tripinfo)
            except: pass

# --- GA 參數設定與初始化 ---
POP_SIZE = 100 
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
    # 👑 【修正】CSV 標題新增 車禍 與 死鎖 欄位
    csv_writer.writerow(["generation", "phase1", "phase2", "total_score", "collisions", "deadlocks", "os_pid"])

    pop = toolbox.population(n=POP_SIZE)
    first_values = 0
    hof = tools.HallOfFame(1) 

    print(f"\n🔁 開始進行 GA 訓練...\n")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(f"\n🔁 開始評估初始群體 (第 0 代)，共 {POP_SIZE} 個體 (多核心加速中...)\n")
        
        # 👑 【核心修正】：攔截初始群體的多重輸出
        results = list(executor.map(toolbox.evaluate, pop))
        for ind, res in zip(pop, results):
            ind.fitness.values = (res[0],) # 只把總分塞給 GA
            ind.col_cnt = res[1]           # 掛上車禍名牌
            ind.dl_cnt = res[2]            # 掛上死鎖名牌
            
        hof.update(pop)
        first_values = hof[0].fitness.values[0] 
            
        print(f"✅ Gen 0 初始群體評估完成！\n")
        PATIENCE = 10  
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

            pop[:] = offspring
            hof.update(pop)
            
            current_gen_best = tools.selBest(pop, k=1)[0] 
            csv_file.flush()
            global_best = hof[0]                          

            # 👑 印出時加上名牌資訊
            print(f"第 {gen+1} 代 當代最佳：{current_gen_best}, 總分: {current_gen_best.fitness.values[0]:.2f} (車禍:{current_gen_best.col_cnt}, 死鎖:{current_gen_best.dl_cnt})")
            print(f"第 {gen+1} 代 歷史最佳：{global_best}, 總分: {global_best.fitness.values[0]:.2f} (車禍:{global_best.col_cnt}, 死鎖:{global_best.dl_cnt})")

            current_best_fit = hof[0].fitness.values[0]
            # 👑 【修正】寫入 CSV 時，把名牌資訊也寫進去
            csv_writer.writerow([gen + 1, current_gen_best[0], current_gen_best[1], f"{current_gen_best.fitness.values[0]:.2f}", current_gen_best.col_cnt, current_gen_best.dl_cnt, f"{os.getpid()}"])
            csv_file.flush()
            
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