# 如何啟動 (Command Line)

請務必在執行 RL_controller.py 時，傳入一個唯一的 ID，例如：

Bash

## 第一次運行/開始新的訓練

python RL_controller.py run_A

## 第二個訓練或多進程中的第二個實例

python RL_controller.py run_B

## 載入 run_A 的進度並繼續訓練

python RL_controller.py run_A
1. 訓練模式 (Train Mode)
如果您想訓練一個名為 my_new_model 的模型：

Bash

## 語法: python3 RL_controller.py train <模型名稱>
python3 ./RL_controller.py train my_new_model 2> ./error_train.txt > ./execute_train.txt 
is_train_mode 會被設為 True。

instance_id 會是 my_new_model。

模型檔案會儲存為 model_my_new_model.h5 和 target_model_my_new_model.h5。

2. 測試模式 (Test Mode)
如果您想用剛訓練好的 my_new_model 進行測試：

Bash

## 語法: python3 RL_controller.py test <模型名稱>
python3 ./RL_controller.py test my_new_model 2> ./error_test.txt > ./execute_test.txt
is_train_mode 會被設為 False。

程式會嘗試載入 model_my_new_model.h5。

agent.exploration_rate 會被設定為 0.0。

3. 互動模式 (如果沒有提供參數)
如果您只輸入 python3 ./RL_controller.py，程式會進入互動式提示，要求您輸入模式和模型名稱。

+ Reference by 冷淡的真實對話師

