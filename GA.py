import traci
from traci._trafficlight import Logic, Phase
from plyer import notification 
import xml.etree.ElementTree as ET
import concurrent.futures # 【新增】用於多核心並行處理
import os
import sys
from deap import base, creator, tools
import random
import csv
import datetime

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
GA_INSTANCE_ID = "default_ga"
if len(sys.argv) > 1:
    GA_INSTANCE_ID = sys.argv[1]
print(f"啟動 GA 實例 ID: {GA_INSTANCE_ID}")

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = rf"./GA_{GA_INSTANCE_ID}__{timestamp}.csv"

csv_file = open(file=filename, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["generation", "phase1", "phase2", "delay"])

TRIPINFO_OUTPUT_PATH = f"tripinfo_{GA_INSTANCE_ID}.xml"

sumoCmd = [
    "sumo", 
    "-c", "osm.sumocfg",
    "--time-to-teleport", "-1",
    "--tripinfo-output", TRIPINFO_OUTPUT_PATH
]

def get_total_delay(filename):
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
    except (FileNotFoundError, ET.ParseError) as e:
        print(f"警告：無法解析或讀取 '{filename}' (錯誤: {e}). 返回極大延遲作為懲罰。", file=sys.stderr)
        return 1e9
        
    total_waiting_time = 0.0
    for trip in root.findall("tripinfo"):
        if "timeLoss" in trip.attrib:
            timeLoss = float(trip.attrib["timeLoss"])
            total_waiting_time += timeLoss
    return total_waiting_time

# 【修正：將 evaluate 函數的 tripinfo 檔案名改為動態，以支援並行】
def evaluate(individual):
    # 【關鍵修正 3】：使用 Process ID 來創建獨立的 tripinfo 檔案和 TraCI label
    pid = os.getpid()
    
    # 確保每個進程的輸出檔案和 TraCI 連線名稱都是唯一的
    unique_tripinfo = f"tripinfo_{GA_INSTANCE_ID}_PID{pid}.xml"
    unique_sumo_cmd = [
        "sumo", 
        "-c", "osm.sumocfg",
        "--time-to-teleport", "-1",
        "--tripinfo-output", unique_tripinfo
    ]

    try:
        # 使用唯一的 label 啟動 TraCI
        traci.start(unique_sumo_cmd, label=f"GA_TL_{pid}") 

        # --- 建立時相邏輯 ---
        logic = Logic(
            programID="ga_prog",
            phases=[            
                Phase(individual[0], 'G' * 12 + 'r' * 12),
                Phase(3, 'y' * 12 + 'r' * 12),
                Phase(individual[1], 'r' * 12 + 'G' * 12), 
                Phase(3, 'r' * 12 + 'y' * 12)
            ],
            type=0,
            currentPhaseIndex=0
        )
        traci.trafficlight.setProgramLogic(TRAFFIC_LIGHT_ID, logic)
        traci.trafficlight.setProgram(TRAFFIC_LIGHT_ID, logic.programID)
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 0)
        
        # 確保模擬運行足夠長的時間
        MAX_SIM_STEPS = 600
        step = 0
        while step < MAX_SIM_STEPS and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
        
        # 獲取總延遲
        try:
            delay = get_total_delay(unique_tripinfo) 
            return delay,
        except Exception as xml_e:
            return (1e9,) 
            
    except traci.TraCIException as e:
        return (1e9,) 
    except Exception as e_general:
        return (1e9,) 
    finally:
        try:
             traci.close()
             # 模擬結束後刪除臨時 tripinfo 檔案
             if os.path.exists(unique_tripinfo):
                 os.remove(unique_tripinfo)
        except Exception:
             pass

# --- GA 參數設定與初始化 (保持不變) ---
POP_SIZE = 100
GEN_NUM = 100
TIME_MIN = 5
TIME_MAX = 100

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, TIME_MIN, TIME_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate) # 使用修正後的 evaluate 函數
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=TIME_MIN, up=TIME_MAX, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)


# --- 執行 GA 訓練 (使用 ProcessPoolExecutor) ---
pop = toolbox.population(n=POP_SIZE)
first_values = 0

print("\n🔁 開始進行 GA 訓練...\n",flush=True)

# 【關鍵修正 1】：將 ProcessPoolExecutor 放在最外層
with concurrent.futures.ProcessPoolExecutor() as executor:
    
    # 【關鍵修正 2】：使用 executor.map 評估初始族群，並統一賦值
    print(f"\n🔁 開始評估初始群體 (Generation 0)，共 {POP_SIZE} 個體 (多核心加速中...)\n" ,flush=True)
    
    # 這裡的 map 是並行的，但結果是按順序返回的
    fitnesses = list(executor.map(toolbox.evaluate, pop))
    
    # 將適應度賦值給個體
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    print(f"✅ Gen 0 初始群體評估完成！\n", flush=True)

    # --- 主世代迴圈 ---
    for gen in range(GEN_NUM):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 交配、突變 (保持不變)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:   
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.9:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 計算新的適應度 (使用多核心)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print(f"🔄 第 {gen+1} 代：開始評估 {len(invalid_ind)} 個新個體 (多核心加速中...)", flush=True)
        
        # 【關鍵修正 2 續】：使用 executor.map 進行多核心評估
        new_fitnesses = list(executor.map(toolbox.evaluate, invalid_ind))
        
        for ind, fit in zip(invalid_ind, new_fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        best = tools.selBest(pop, 1)[0]

        if gen == 0:
            first_values = best.fitness.values[0]

        print(f"第 {gen+1} 代最佳紅綠燈組合：{best}, 等待時間：{best.fitness.values[0]:.2f} 秒",flush=True)

        # 1. 寫入【完整日誌檔】
        csv_writer.writerow([gen + 1, best[0], best[1], f"{best.fitness.values[0]:.2f}"])
        csv_file.flush()

        # 2. 【即時更新】固定名稱結果檔 (確保中斷也能拿到最好結果)
        FINAL_RESULT_FILENAME = "./GA_best_result.csv" 
        try:
            with open(FINAL_RESULT_FILENAME, mode="w", newline="", encoding="utf-8") as final_f:
                final_writer = csv.writer(final_f)
                final_writer.writerow(["generation", "phase1", "phase2", "delay"])
                final_writer.writerow([
                    gen + 1, 
                    best[0], 
                    best[1], 
                    f"{best.fitness.values[0]:.2f}"
                ])
        except Exception as e:
            print(f"警告：無法寫入最終 GA 結果檔案: {e}")

# --- 輸出最終最佳解 ---
final_best = tools.selBest(pop, 1)[0]
print("\n✅ 訓練完成！")
# ... (後續的輸出和通知邏輯保持不變) ...
# --- 輸出最終最佳解 ---
final_best = tools.selBest(pop, 1)[0]
print("\n✅ 訓練完成！")
print(f"最佳紅綠燈時間組合為：{final_best}")
print(f"總等待時間：{final_best.fitness.values[0]:.2f} 秒")
print(f"第一代等待時間：{first_values:.2f} 秒")
# 【新增：將最終最佳解寫入固定名稱檔案】
FINAL_RESULT_FILENAME = "./GA_best_result.csv" 

try:
    with open(FINAL_RESULT_FILENAME, mode="w", newline="", encoding="utf-8") as final_f:
        final_writer = csv.writer(final_f)
        # 僅寫入標頭和最佳結果
        final_writer.writerow(["generation", "phase1", "phase2", "delay"])
        final_writer.writerow([
            GEN_NUM, 
            final_best[0], 
            final_best[1], 
            f"{final_best.fitness.values[0]:.2f}"
        ])
    print(f"📄 已將最終最佳解寫入固定檔案 {FINAL_RESULT_FILENAME}")
except Exception as e:
    print(f"警告：無法寫入最終 GA 結果檔案: {e}")

    notification.notify(
    title = "Python GA Trainning Finish",
    message = f"RUN PID: {os.getpid()} , MODEL ID= GA {timestamp}" ,
        
    # displaying time
    timeout=10 # seconds
)
# 關閉 csv 檔
csv_file.close()
print(f"\n📄 已將所有結果寫入 {filename}")