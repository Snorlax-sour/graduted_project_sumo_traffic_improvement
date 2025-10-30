import random
import numpy as np # pyright: ignore[reportMissingImports]
from collections import deque
from tensorflow.keras.models import Sequential, load_model # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, Input # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]
from tensorflow.keras.metrics import MeanSquaredError
 # <--- 【新增或確認】
import os # 新增：用於檢查檔案是否存在

class DQNAgent:
    # --- 【修正點 1：新增 instance_id 參數】---
    def __init__(self, state_size, action_space, instance_id="default_rl"):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.memory = deque(maxlen=2000)

        # 超參數 - 命名已統一
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.999
        self.learning_rate = 0.0001
        self.update_target_freq = 20 # 讓目標網路更新的頻率稍微降低
        self.train_counter = 0

        # --- 檔案名稱設定 (使用 ID 隔離) ---
        self.instance_id = instance_id
        self.model_filename = f"model_{self.instance_id}.h5"
        self.target_model_filename = f"target_model_{self.instance_id}.h5"

        # 【修正】: 將模型初始化為 None，延遲建立
        self.model = None
        self.target_model = None

    def build_models(self):
        """根據 self.state_size 建立主網路和目標網路。"""
        if self.model is None: # 只有在模型尚未建立時才建立
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
            # 【修正點 A：編譯時使用物件而非字串】
        # 建立主網路
        # self.model.compile(
        #     loss=MeanSquaredError(), # <--- 將 'mse' 字串替換為 MeanSquaredError() 實例
        #     optimizer=Adam(learning_rate=self.learning_rate)
        # )

        # # 建立目標網路
        # self.target_model.compile(
        #     loss=MeanSquaredError(), # <--- 將 'mse' 字串替換為 MeanSquaredError() 實例
        #     optimizer=Adam(learning_rate=self.learning_rate)
        # )

    def _build_model(self):
        # Keras 推薦的新寫法，避免 UserWarning
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=['mse'])
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        # 【修正】: 確保模型已建立
        if self.model is None:
            raise RuntimeError("模型尚未建立。請在初始化 Agent 後呼叫 build_models()。")

        if np.random.rand() <= self.exploration_rate:
            return random.choice(self.action_space)
        
        state_tensor = np.array([state])
        act_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(act_values[0])

    # def replay(self, batch_size):
    #     if len(self.memory) < batch_size:
    #         return
            
    #     minibatch = random.sample(self.memory, batch_size)
    #     # --- 【優化點：將樣本轉換為批次陣列】 ---
    #     # 1. 從記憶庫中分離出所有元素，並轉換為 NumPy 陣列
    #     # *minibatch: 將 list of tuples 展開
    #     # map(np.array, zip(...)): 高效地將所有元素打包成獨立的 NumPy 陣列
    #     states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
    #     for state, action, reward, next_state, done in minibatch:
    #         state_tensor = np.array([state])
    #         next_state_tensor = np.array([next_state])

    #         target = reward
    #         if not done:
    #             # 使用 target_model 來預測未來收益 (Q-value)
    #             target = (reward + self.discount_factor * 
    #                       np.amax(self.target_model.predict(next_state_tensor, verbose=0)[0]))
            
    #         # 使用 model 來獲取當前的預測，並只更新採取了的 action 對應的分數
    #         target_f = self.model.predict(state_tensor, verbose=0)
    #         target_f[0][action] = target
            
    #         self.model.fit(state_tensor, target_f, epochs=1, verbose=0)

    #     # (將所有學習後的更新都放在 replay 中)
    #     self.train_counter += 1
    #     if self.train_counter % self.update_target_freq == 0:
    #         self.update_target_model()
    #         print(f"*** 目標網路已在第 {self.train_counter} 步訓練後更新 ***")
        
    #     if self.exploration_rate > self.min_exploration:
    #         self.exploration_rate *= self.exploration_decay
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        if self.model is None: # 增加安全檢查
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        # --- 【優化點：將樣本轉換為批次陣列】 ---
        # 1. 從記憶庫中分離出所有元素，並轉換為 NumPy 陣列
        # *minibatch: 將 list of tuples 展開
        # map(np.array, zip(...)): 高效地將所有元素打包成獨立的 NumPy 陣列
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        # 2. 【單次呼叫】使用 Target Model 預測所有下一狀態的 Q 值
        # target_q_next 的 shape: (batch_size, action_size)
        target_q_next = self.target_model.predict(next_states, verbose=0)
        
        # 3. 計算 DQN 的 Target Q 值
        # 找出每個 next_state 的最大 Q 值 (np.amax(..., axis=1))
        # dones (布林值) 轉換為 (0 或 1) 後，用於判斷是否要加上未來獎勵
        q_max = np.amax(target_q_next, axis=1) 
        
        # target = reward + discount_factor * max_Q(S') * (1 - done)
        targets = rewards + self.discount_factor * (1 - dones) * q_max
        
        # 4. 【單次呼叫】使用 Main Model 獲取當前所有狀態的 Q 值預測
        # target_f 的 shape: (batch_size, action_size)
        target_f = self.model.predict(states, verbose=0)
        
        # 5. 僅更新實際採取的 action 對應的 Q 值
        # np.arange(batch_size) 產生 [0, 1, 2, ...] 的索引
        # actions 陣列提供要更新的欄位索引
        batch_indices = np.arange(batch_size)
        target_f[batch_indices, actions] = targets
        
        # 6. 【單次呼叫】訓練整個批次
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        # ---------------------------------------------

        # (將所有學習後的更新都放在 replay 中)
        self.train_counter += 1
        if self.train_counter % self.update_target_freq == 0:
            self.update_target_model()
            print(f"*** 目標網路已在第 {self.train_counter} 步訓練後更新 ***")
        
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay
    def learn(self, state, action, reward, next_state):
        self.remember(state, action, reward, next_state, False)
        # 在記憶庫足夠大時才開始學習
        if len(self.memory) > 32: 
            self.replay(batch_size=32)

    # --- 【修正點 2：新增儲存模型的方法】---
    def save_model(self, filename="model_weights.h5"):
        """將主網路和目標網路模型儲存到帶有唯一 ID 的檔案"""
        try:
            self.model.save(self.model_filename)
            self.target_model.save(self.target_model_filename)
            print(f"\n✅ RL 模型已儲存: {self.model_filename}")
        except Exception as e:
            print(f"\n❌ 模型儲存失敗: {e}")

    # --- 【修正點 3：新增載入模型的方法】---
    def load_model(self):
        """從帶有唯一 ID 的檔案載入模型"""
        # 確保 model 檔案存在
        if not os.path.exists(self.model_filename):
            print(f"警告：找不到模型檔案 '{self.model_filename}'，將從頭開始訓練。")
            return False
            
        try:
            # 【修正】: 載入模型前，先確保本地模型結構已建立
            if self.model is None:
                self.build_models()
            # 【關鍵修正】：使用 custom_objects 參數解決反序列化錯誤
            custom_objects = {
                # 將模型儲存時使用的 'mse' 字串映射到實際的 Keras 函式
                'mse': MeanSquaredError,
                'mean_squared_error': MeanSquaredError,
                # 雖然 Adam 通常會自動載入，但明確指定更安全
                'Adam': Adam 
            }
            self.model = load_model(self.model_filename, 
                custom_objects=custom_objects, # <--- 修正點
                compile=True # 確保模型載入後是可用的
                )
            self.target_model = load_model(self.target_model_filename, 
                custom_objects=custom_objects, # <--- 修正點
                compile=True # 確保模型載入後是可用的
                )
            self.update_target_model() # 載入後同步權重
            
            # 載入模型後，將探索率降到最低
            self.exploration_rate = self.min_exploration
            print(f"✅ RL 模型已載入: {self.model_filename}. Epsilon 重設為 {self.min_exploration}")
            return True
        except Exception as e:
            # 如果載入失敗，則忽略並使用新模型
            print(f"❌ 無法載入模型 '{self.model_filename}' (錯誤: {e})，將從頭開始訓練。")
            return False

    