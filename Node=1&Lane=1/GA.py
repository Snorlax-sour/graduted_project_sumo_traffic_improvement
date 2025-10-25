import traci
from traci._trafficlight import Logic, Phase

import xml.etree.ElementTree as ET

import os

from deap import base, creator, tools
import random

import csv
import datetime

# 存入當下的時間
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = rf"C:\Users\user\Desktop\course\專題\software\Node=1&Lane=1\GA_result\GA_{timestamp}.csv"

# 寫入指定的csv檔案
csv_file = open(file=filename, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["generation", "phase1", "phase2", "delay"])

# 啟動 SUMO 模擬（用 sumo-gui 可視化，或 sumo 為命令列）
sumoCmd = ["sumo", "-c", "test.sumocfg"]

# 取得總等待時間
def get_total_delay(filename="tripinfo.xml"):
    tree = ET.parse(filename)                       # 載入整個 XML 樹狀結構
    root = tree.getroot()                           # 取得 XML 的根元素（整個 <tripinfos> 標籤）
    total_waiting_time = 0.0
    for trip in root.findall("tripinfo"):           # 對每一個 <tripinfo> 標籤（代表一輛車）進行迴圈
        timeLoss = float(trip.attrib["timeLoss"])   # 從 <tripinfo> 的 timeLoss 屬性抓出來（是字串），轉成 float。
        total_waiting_time += timeLoss
    return total_waiting_time

# GA 評估函數
def evaluate(ind):
    # 東西向綠燈時間, 南北向綠燈時間 = 個體[phase1, phase2]
    phase1, phase2 = ind

    # 如果有上一次的紀錄, 就刪除舊的紀錄
    if os.path.exists("tripinfo.xml"):
        os.remove("tripinfo.xml")

    # 開始模擬
    traci.start(sumoCmd)

    # 取得路口n4
    tl_id = traci.trafficlight.getIDList()[0]

    # 路口參數
    logic = Logic(
        programID="ga-program",                     # 紅綠燈邏輯的名稱
        type=0,                                     # 類型（0 = 固定邏輯）
        currentPhaseIndex=0,                        # 初始相位 index（從哪一個 phase 開始）
        phases=[                                    # 相位列表
            Phase(phase1, "GGgrrrGGgrrr", 0, 0),
            Phase(5, "yyyrrryyyrrr", 0, 0),
            Phase(phase2, "rrrGGgrrrGGg", 0, 0),
            Phase(5, "rrryyyrrryyy", 0, 0)
        ]
    )
    traci.trafficlight.setProgramLogic(tl_id, logic)

    # 跑 500 ms
    for step in range(500):
        traci.simulationStep()

    # 結束模擬
    traci.close()

    # 呼叫自定義函數取得總等待時間
    delay = get_total_delay("tripinfo.xml")
    return (delay,) # tuple

# --- GA 參數設定 ---
POP_SIZE = 100          # 每代要訓練幾組個體（幾組紅綠燈設定）
GEN_NUM = 100           # 總共進化幾代
TIME_MIN = 5            # 綠燈最短秒數
TIME_MAX = 100          # 綠燈最長秒數

# --- GA 初始化 ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 越小越好
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
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
first_values = 0

print("\n🔁 開始進行 GA 訓練...\n")

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
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    best = tools.selBest(pop, 1)[0]

    if gen == 0:
        first_values = best.fitness.values[0]

    print(f"第 {gen+1} 代最佳紅綠燈組合：{best}, 等待時間：{best.fitness.values[0]:.2f} 秒")

    # 將 ["generation", "phase1", "phase2", "delay"] 存入csv
    csv_writer.writerow([gen + 1, best[0], best[1], f"{best.fitness.values[0]:.2f}"])

# --- 輸出最終最佳解 ---
final_best = tools.selBest(pop, 1)[0]
print("\n✅ 訓練完成！")
print(f"最佳紅綠燈時間組合為：{final_best}")
print(f"總等待時間：{final_best.fitness.values[0]:.2f} 秒")
print(f"第一代等待時間：{first_values:.2f} 秒")

# 關閉 csv 檔
csv_file.close()
print(f"\n📄 已將所有結果寫入 {filename}")