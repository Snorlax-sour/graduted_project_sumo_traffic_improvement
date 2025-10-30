import traci
from traci._trafficlight import Logic, Phase
from plyer import notification # <--- 新增
import xml.etree.ElementTree as ET

import os
import sys
from deap import base, creator, tools
import random
def get_sumo_home():
    """找到 SUMO 的安装目录并設定環境"""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        return True
    else:
        sys.exit("請確認 SUMO_HOME 環境變數已設定！")
get_sumo_home()

import csv
import datetime
# 與RL_controller同一個路口設定
TRAFFIC_LIGHT_ID="1253678773"

# --- 【新增：從命令列讀取唯一 ID】---
GA_INSTANCE_ID = "default_ga"
if len(sys.argv) > 1:
    GA_INSTANCE_ID = sys.argv[1]
print(f"啟動 GA 實例 ID: {GA_INSTANCE_ID}")

# 存入當下的時間
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S") # <-- 讓這行生效
filename = rf"./GA_{GA_INSTANCE_ID}__{timestamp}.csv" # <-- 導向一個帶時間戳的專屬檔案
# filename = rf"./GA_best_result.csv" # <-- 舊的這行註釋掉或移除



# 寫入指定的csv檔案
csv_file = open(file=filename, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["generation", "phase1", "phase2", "delay"])
# --- 【修正：讓 tripinfo.xml 帶有 ID】---
TRIPINFO_OUTPUT_PATH = f"tripinfo_{GA_INSTANCE_ID}.xml"

# 啟動 SUMO 模擬（用 sumo-gui 可視化，或 sumo 為命令列）
sumoCmd = [
    "sumo", 
    "-c", "osm.sumocfg",
    "--time-to-teleport", "-1", # 避免車輛瞬移
    "--tripinfo-output", TRIPINFO_OUTPUT_PATH # 確保輸出 tripinfo.xml
]

# 取得總等待時間
# 取得總等待時間
def get_total_delay(filename="tripinfo.xml"):
    # 【修正：增加 XML 檔案讀取和解析的錯誤處理】
    try:
        tree = ET.parse(filename)                       # 載入整個 XML 樹狀結構
        root = tree.getroot()                           # 取得 XML 的根元素（整個 <tripinfos> 標籤）
    except (FileNotFoundError, ET.ParseError) as e:
        # 如果檔案不存在或解析失敗 (unclosed token)，返回極大懲罰值
        print(f"警告：無法解析或讀取 '{filename}' (錯誤: {e}). 返回極大延遲作為懲罰。", file=sys.stderr)
        return 1e9 # 返回一個極大的值作為懲罰
        
    total_waiting_time = 0.0
    for trip in root.findall("tripinfo"):
        # 確保屬性存在，以免 tripinfo.xml 雖然解析成功但內容不完整
        if "timeLoss" in trip.attrib:
            timeLoss = float(trip.attrib["timeLoss"])
            total_waiting_time += timeLoss
    return total_waiting_time

# context: GA.py 檔案中

def evaluate(individual):
    # 【修正：使用 try...finally 確保 traci 關閉】
    try:
        # 啟動 SUMO
        # 注意: 這裡的 label=TRAFFIC_LIGHT_ID 避免多個 traci 實例衝突
        traci.start(sumoCmd, label=TRAFFIC_LIGHT_ID) 

        # --- 建立時相邏輯 (保持不變) ---
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
        # 3. 模擬結束，關閉連線
        # traci.close()
        # 獲取總延遲 (現在有錯誤處理，更安全)
        # 【保護層 2：tripinfo.xml 讀取防護】
        try:
            delay = get_total_delay("tripinfo.xml") 
        
            return delay, # 必須回傳一個 tuple
        except Exception as xml_e:
            # 如果 XML 檔案讀取失敗 (檔案損壞/不完整)
            print(f"警告：無法解析或讀取 'tripinfo.xml' (錯誤: {xml_e}). 返回極大延遲作為懲罰。")
            return (1e9,) # 返回懲罰值
    except traci.TraCIException as e:
        print(f"🚨 嚴重警告：SUMO/TraCI 連線中斷 (錯誤: {e}). 對當前個體返回極大懲罰。")
        # 嘗試關閉連線，避免佔用埠口
        if traci.is_connected():
            traci.close()
            
        return (1e9,) # 返回懲罰值
    except Exception as e_general:
        # 捕獲所有其他未預期的錯誤
        print(f"🚨 致命錯誤：在評估過程中發生未預期錯誤 (錯誤: {e_general}). 對當前個體返回極大懲罰。")
        if traci.is_connected():
            traci.close()
        return (1e9,) # 返回懲罰值
    finally:
        # 確保在任何情況下都關閉 traci 連線
        # 【關鍵修正】安全關閉 TraCI，避免 is_connected() 錯誤
        try:
             traci.close()
        except Exception:
             # 如果 TraCI 已經因為崩潰而斷開，traci.close() 會報錯，忽略即可
             pass

# --- GA 參數設定 ---
POP_SIZE = 100          # 每代要訓練幾組個體（幾組紅綠燈設定）
GEN_NUM = 100           # 總共進化幾代
TIME_MIN = 5            # 綠燈最短秒數
TIME_MAX = 100          # 綠燈最長秒數

# --- GA 初始化 ---
# --- GA 初始化 【最終修正點：加入 hasattr 檢查】 ---
# 1. 檢查 FitnessMin
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 越小越好

# 2. 檢查 Individual
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, TIME_MIN, TIME_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)  # 2個phase
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=TIME_MIN, up=TIME_MAX, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- 執行 GA 訓練 ---
pop = toolbox.population(n=POP_SIZE)
fitnesses = list(map(toolbox.evaluate, pop))
# for ind, fit in zip(pop, fitnesses):
#     ind.fitness.values = fit
first_values = 0

print("\n🔁 開始進行 GA 訓練...\n",flush=True)
print(f"\n🔁 開始評估初始群體 (Generation 0)，共 {POP_SIZE} 個體...\n" ,flush=True)

# 【關鍵修正：使用迴圈評估初始群體並即時輸出】
fitnesses = []
for i, individual in enumerate(pop):
    fit = toolbox.evaluate(individual)
    fitnesses.append(fit)
    
    # 這裡會即時印出進度！
    print(f"✅ Gen 0 完成個體 {i + 1}/{POP_SIZE} 評估. 延遲 (Penalty): {fit[0]:.2f} 秒。",flush=True)
    
# 將適應度賦值給個體
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit


for gen in range(GEN_NUM):
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # 交配
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.8:   
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    # 突變
    for mutant in offspring:
        if random.random() < 0.9:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 計算新的適應度
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    # fitnesses = list(map(toolbox.evaluate, invalid_ind))
    fitnesses = []
    print(f"🔄 第 {gen+1} 代：開始評估 {len(invalid_ind)} 個新個體...", flush=True)
    for i, ind in enumerate(invalid_ind):
        fit = toolbox.evaluate(ind)
        fitnesses.append(fit)
        print(f"   -> 完成 {i+1}/{len(invalid_ind)} 個體評估. 延遲: {fit[0]:.2f} 秒。", flush=True)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    best = tools.selBest(pop, 1)[0]

    if gen == 0:
        first_values = best.fitness.values[0]

    print(f"第 {gen+1} 代最佳紅綠燈組合：{best}, 等待時間：{best.fitness.values[0]:.2f} 秒",flush=True)

    # 將 ["generation", "phase1", "phase2", "delay"] 存入csv
    csv_writer.writerow([gen + 1, best[0], best[1], f"{best.fitness.values[0]:.2f}"])
    # 【新增：強制將緩衝區資料寫入磁碟】
    csv_file.flush() # 這一行很關鍵！

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