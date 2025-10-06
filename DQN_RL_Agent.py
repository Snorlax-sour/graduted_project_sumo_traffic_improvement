import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_space):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.memory = deque(maxlen=2000)

        # 超參數 - 命名已統一
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.999
        self.learning_rate = 0.001
        self.update_target_freq = 20 # 讓目標網路更新的頻率稍微降低
        self.train_counter = 0

        # 建立主網路和目標網路
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Keras 推薦的新寫法，避免 UserWarning
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.choice(self.action_space)
        
        state_tensor = np.array([state])
        act_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = np.array([state])
            next_state_tensor = np.array([next_state])

            target = reward
            if not done:
                # 使用 target_model 來預測未來收益 (Q-value)
                target = (reward + self.discount_factor * 
                          np.amax(self.target_model.predict(next_state_tensor, verbose=0)[0]))
            
            # 使用 model 來獲取當前的預測，並只更新採取了的 action 對應的分數
            target_f = self.model.predict(state_tensor, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state_tensor, target_f, epochs=1, verbose=0)

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