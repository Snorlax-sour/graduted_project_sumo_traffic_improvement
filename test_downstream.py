import traci
import sumo_utils
TRAFFIC_LIGHT_ID = "1253678773"
SUMO_CONFIG_FILE = "osm.sumocfg"
ACTION_SPACE = [0, 1]

print("🕵️‍♂️ 正在派遣偵察兵探勘 SUMO 實際路口特徵...")
temp_sumo_cmd = sumo_utils.build_sumo_cmd(
    config_file=SUMO_CONFIG_FILE,
    use_gui=False,       # 隱形模式，不開 GUI
    tripinfo_file=None,  # 探勘不需要產出 XML
    seed=42,
    quiet=True           # 安靜模式，不噴多餘的 Log
)
step = 0
MAX_SIMULATION_STEPS = 8000
traci.start(temp_sumo_cmd)
while step < MAX_SIMULATION_STEPS:
    sumo_utils.check_downstream_jam(TRAFFIC_LIGHT_ID,debugging=True)
    step += 1
    traci.simulationStep()
    # print(f"step: {step}")
traci.close()