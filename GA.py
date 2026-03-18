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
# 使用環境變數鎖定啟動時間，確保多核心 Worker 不會因為秒差而建立出不同的檔案
if "GA_START_TIME" not in os.environ:
    os.environ["GA_START_TIME"] = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

LOG_FILENAME = f"execute_GA_{os.environ['GA_START_TIME']}.txt"

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        # 使用 "a" (append) 模式，讓多核心進程可以共同寫入同一個檔案
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 強制即時寫入，避免多核心同時寫入時發生卡彈

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 將系統的標準輸出與錯誤輸出替換為我們的 DualLogger
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

def get_total_delay(filename):
    """解析 XML 計算延遲。如果出錯，返回懲罰值。"""
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
    except (FileNotFoundError, ET.ParseError) as e:
        print(f"[{os.getpid()}] 警告：無法解析 XML '{filename}' (錯誤: {e}). 返回極大延遲作為懲罰。")
        return 999999.0 # 給予極大的懲罰值，讓 GA 淘汰這個壞個體
        
    total_waiting_time = 0.0
    for trip in root.findall("tripinfo"):
        if "timeLoss" in trip.attrib:
            timeLoss = float(trip.attrib["timeLoss"])
            total_waiting_time += timeLoss
    return total_waiting_time


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

        # ... 紅綠燈設定邏輯（保持不變）...
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
        
        # ===== 👇 關鍵替換：使用 sumo_utils 👇 =====
        total_deadlock_penalty = 0.0
        total_collision_count = 0
        MAX_SIM_STEPS = 8000
        step = 0
        
        while step < MAX_SIM_STEPS and conn.simulation.getMinExpectedNumber() > 0:
            conn.simulationStep()
            step += 1
            
            # 👑 使用共用函數：碰撞偵測
            new_collisions = sumo_utils.detect_real_collisions(
                TRAFFIC_LIGHT_ID, 
                active_crashes, 
                step,
                conn=conn  # 👈 傳入 conn
            )
            total_collision_count += new_collisions
            
            # 👑 使用共用函數：更新碰撞車輛
            sumo_utils.update_crash_vehicles(active_crashes, conn=conn)
            
            # 👑 使用共用函數：死鎖處理
            deadlock_penalty = sumo_utils.handle_deadlock_vehicles(
                deadlock_threshold=90,
                conn=conn  # 👈 傳入 conn
            )
            total_deadlock_penalty += deadlock_penalty
            
            # 心跳監視器
            if step % 500 == 0:
                print(f"[PID {pid}] 正在執行模擬... 第 {step}/{MAX_SIM_STEPS} 步 (剩餘車輛: {conn.simulation.getMinExpectedNumber()})")

        # ===== 👆 替換結束 👆 =====

        conn.close()
        
        # 👑 使用共用函數：計算延遲
        delay = sumo_utils.calculate_delay(TRAFFIC_LIGHT_ID, conn=conn)
        
        # 注意：這裡的 total_deadlock_penalty 已經是負數了
        # 所以要改成「減去」才能增加懲罰
        final_penalty_score = delay - total_deadlock_penalty + (total_collision_count * 1000.0)
        
        return (final_penalty_score,)
            
    except Exception as e:
        print(f"Error in PID {pid}: {e}")
        return (999999.0,) 
    finally:
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
toolbox.register("select", tools.selTournament, tournsize=2)
    # 每2人一組對打，打100場，1v1


# 【修復 5】：主程式保護！這在 Windows 使用多核心是必備的
def main():
    print(f"主程序 PID {os.getpid()}: 啟動 GA 實例 ID: {GA_INSTANCE_ID}")
    print(f"📝 本次訓練日誌將自動寫入: {LOG_FILENAME}")
    # =========================================================
    # 👑 【新增】讀取跨世代的歷史最佳紀錄
    # =========================================================
    historical_best_delay = float('inf')
    FINAL_RESULT_FILENAME = "./GA_best_result.csv"
    if os.path.exists(FINAL_RESULT_FILENAME):
        try:
            with open(FINAL_RESULT_FILENAME, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None) # 跳過標題列
                row = next(reader, None)
                if row and len(row) >= 4:
                    historical_best_delay = float(row[3])
                    print(f"💾 成功載入歷史最佳紀錄：{historical_best_delay} 秒")
        except Exception as e:
            print(f"⚠️ 讀取歷史紀錄失敗，視為全新的開始: {e}")
    # =========================================================
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

    print(f"\n🔁 開始進行 GA 訓練...\n")

    # 開啟多核心運算
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(f"\n🔁 開始評估初始群體 (第 0 代)，共 {POP_SIZE} 個體 (多核心加速中...)\n")
        
        fitnesses = list(executor.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
        # 【修復 B：更新 Generation 0 到名人堂】
        hof.update(pop)
        first_values = hof[0].fitness.values[0] # 記錄初始群體的全局最佳
            
        print(f"✅ Gen 0 初始群體評估完成！\n")
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
            print(f"🔄 第 {gen+1} 代：開始評估 {len(invalid_ind)} 個新個體 (多核心加速中...)")
            
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

            print(f"第 {gen+1} 代 當代最佳組合：{current_gen_best}, 等待時間：{current_gen_best.fitness.values[0]:.2f} 秒")
            print(f"第 {gen+1} 代 歷史最佳組合：{global_best}, 最小等待時間：{global_best.fitness.values[0]:.2f} 秒")

            # 檢查是否有進步
            current_best_fit = hof[0].fitness.values[0]
            csv_writer.writerow([gen + 1, current_gen_best[0], current_gen_best[1], f"{current_gen_best.fitness.values[0]:.2f}", f"{os.getpid()}"])
            csv_file.flush()
            if current_best_fit < best_fitness_so_far:
                best_fitness_so_far = current_best_fit
                no_improve_count = 0  # 有進步，計數重置
                FINAL_RESULT_FILENAME = "./GA_best_result.csv" 
                
                # 👑 【修改】只有當前成績超越「跨世代歷史紀錄」時，才覆寫檔案！
                if current_best_fit < historical_best_delay:
                    print(f"🎉 突破跨世代歷史紀錄！({historical_best_delay:.2f} 降至 {current_best_fit:.2f})，更新檔案！")
                    historical_best_delay = current_best_fit  # 更新門檻值
                    try:
                        with open(FINAL_RESULT_FILENAME, mode="w", newline="", encoding="utf-8") as final_f:
                            final_writer = csv.writer(final_f)
                            final_writer.writerow(["generation", "phase1", "phase2", "delay", "os_pid"])
                            final_writer.writerow([gen + 1, global_best[0], global_best[1], f"{global_best.fitness.values[0]:.2f}", f"{os.getpid()}"])
                    except Exception as e:
                        print(f"error write best csv file: {e}")
                else:
                    print(f"👍 本次訓練有進步 ({current_best_fit:.2f})，但尚未打破歷史紀錄 ({historical_best_delay:.2f})。")
            else:
                no_improve_count += 1 # 沒進步，耐性扣點
                
            print(f"第 {gen+1} 代，連續未進步：{no_improve_count}/{PATIENCE}")
            
            # 🌟 【加入這一段：打破近親繁殖的僵局】🌟
            # 👑 【修復】：加上 > 0，防止第 5 代破紀錄時 (0 % 4 == 0) 錯誤滅絕王者！
            if no_improve_count > 0 and no_improve_count % 4 == 0:  
                print(f"⚠️ 偵測到基因庫同質化，保留歷史最強，其餘重新隨機生成！")
                # 保留歷史上最強的那一個 (HOF)
                elite = toolbox.clone(hof[0])
                # 把剩下 99 個全部殺掉，重新產生隨機的新基因
                pop = toolbox.population(n=POP_SIZE)
                pop[0] = elite  # 把最強的放回第 0 個位置保底
                
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