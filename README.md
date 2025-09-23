## 授權 (License)

#### 本專案原始程式碼

本專案的原創 Python 程式碼 (包括 `RL_controller.py`, `RL_Agent.py` 等) 採用 **[MIT License](LICENSE)** 進行授權。您可以自由使用、修改和散佈，但需遵守 MIT 授權的條款。

#### 依賴軟體 (Dependencies)

本專案的運作依賴於 **SUMO (Simulation of Urban MObility)** 軟體套件。

*   **SUMO** 是 Eclipse 基金會旗下的開源專案，其本身是在 **Eclipse Public License 2.0 (EPL-2.0)** 授權下提供。
*   本專案**僅透過 TraCI API 呼叫** SUMO 的功能，**並未修改**任何 SUMO 的原始程式碼。
*   使用者在使用本專案前，需自行下載並安裝 SUMO，並應同時遵守其 EPL-2.0 授權條款。