import random # <--- 新增這個 import

class RLAgent:
    def __init__(self, action_space):
        # 初始化智能體
        self.action_space = action_space # 讓智能體知道有哪些動作可以選, e.g., [0, 1]
        self.learning_rate = 0.1         # 學習率 (alpha): 每次學習的幅度有多大
        self.discount_factor = 0.9       # 折扣因子 (gamma): 對未來的獎勵有多重視
        self.exploration_rate = 1.0      # 探索率 (epsilon): 一開始有 100% 的機率亂試
        self.exploration_decay = 0.995   # 探索率衰減: 每次學習後，減少亂試的機率
        self.min_exploration = 0.05      # 最小探索率: 確保它永遠有機會嘗試新東西
        # 提高最小探索率

        # Q-table: 智能體的「大腦」或「記憶筆記本」
        # 我們用一個字典來儲存，格式是: { 狀態: [動作0的分數, 動作1的分數, ...] }
        self.q_table = {}

    def choose_action(self, state):
        state = str(state) # 將 state (tuple) 轉為字串，使其可以作為字典的 key
        # 根據當前狀態，決定下一步要做什麼動作 (探索 vs. 利用)
        if random.uniform(0, 1) < self.exploration_rate:
            # 隨機探索：隨便選一個動作
            return random.choice(self.action_space)
        else:
            # 利用經驗：選擇當前已知分數最高的動作
            # .get(state, [0] * len(self.action_space)) 的意思是:
            # "如果這個狀態從沒見過，就預設所有動作分數都是0"
            q_values = self.q_table.get(state, [0] * len(self.action_space))
            return q_values.index(max(q_values))

    def learn(self, state, action, reward, next_state):
        # 智能體學習的核心 (更新 Q-table)
        state = str(state)
        next_state = str(next_state)
        # 獲取舊的分數
        old_q_values = self.q_table.get(state, [0] * len(self.action_space))
        old_value = old_q_values[action]

        # 計算下一個狀態能得到的最好分數是多少
        next_max = max(self.q_table.get(next_state, [0.0] * len(self.action_space)))
        
        # 這是 Q-Learning 的核心公式
        # 新分數 = (1 - 學習率) * 舊分數 + 學習率 * (獎勵 + 折扣因子 * 未來的最好分數)
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)

        # 更新 Q-table 裡的筆記
        updated_q_values = old_q_values[:]
        updated_q_values[action] = new_value
        self.q_table[state] = updated_q_values
        
        # 學習完後，稍微降低下次亂試的機率
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay