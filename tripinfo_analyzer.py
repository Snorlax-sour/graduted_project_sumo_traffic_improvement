import xml.etree.ElementTree as ET
import os
import argparse
import glob
from datetime import datetime

def analyze_tripinfo(xml_filepath, deadlocks=0, collisions=0):
    """
    解析單一 SUMO 的 tripinfo.xml 並自動產出對應檔名的 TXT 報告。
    """
    if not os.path.exists(xml_filepath):
        print(f"[警告] 找不到指定的 XML 檔案: {xml_filepath}，已跳過。")
        return False

    # 自動生成輸出檔名 (例如: trip_01.xml -> trip_01_analysis.txt)
    base_name = os.path.splitext(xml_filepath)[0]
    output_txt_filepath = f"{base_name}_analysis.txt"

    # 初始化統計變數
    total_vehicles = 0
    total_time_loss = 0.0
    total_waiting_time = 0.0
    total_duration = 0.0

    try:
        # 解析 XML 結構
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        # 遍歷每一台通過的車輛 (<tripinfo> 標籤)
        for trip in root.findall('tripinfo'):
            total_vehicles += 1
            total_time_loss += float(trip.get('timeLoss', 0))
            total_waiting_time += float(trip.get('waitingTime', 0))
            total_duration += float(trip.get('duration', 0))

        # 計算平均值 (防呆：避免除以 0)
        if total_vehicles > 0:
            avg_time_loss = total_time_loss / total_vehicles
            avg_waiting_time = total_waiting_time / total_vehicles
            avg_duration = total_duration / total_vehicles
        else:
            avg_time_loss = avg_waiting_time = avg_duration = 0.0

        # ==========================================
        # 準備寫入 TXT 報告的內容
        # ==========================================
        report_lines = [
            "==================================================",
            "           🚦 SUMO 交通模擬效能分析報告 🚦           ",
            "==================================================",
            f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析來源: {os.path.abspath(xml_filepath)}",
            "--------------------------------------------------",
            "【1. 基礎流量與安全指標】",
            f" 🚗 總通過車輛數 (Throughput)    : {total_vehicles} 輛",
            f" 💥 總車禍次數   (Collisions)    : {collisions} 次",
            f" 🔒 總死鎖次數   (Deadlocks)     : {deadlocks} 次",
            "",
            "【2. 關鍵時間指標 (所有車輛平均)】",
            f" ⏱️ 實際平均延遲 (Avg TimeLoss)   : {avg_time_loss:.2f} 秒",
            f" 🛑 絕對靜止時間 (Avg WaitingTime): {avg_waiting_time:.2f} 秒",
            f" 🛣️ 總旅行時間   (Avg Duration)   : {avg_duration:.2f} 秒",
            "=================================================="
        ]

        # 寫入 TXT 檔案
        with open(output_txt_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
            
        print(f"✅ 成功: {xml_filepath} -> {output_txt_filepath}")
        print(f"   [表現] 延遲: {avg_time_loss:.2f}s | 車禍: {collisions} | 死鎖: {deadlocks}\n")
        return True

    except Exception as e:
        print(f"[錯誤] 解析 {xml_filepath} 時發生問題: {str(e)}\n")
        return False

def main():
    # 建立命令列參數解析器
    parser = argparse.ArgumentParser(description="SUMO tripinfo.xml 批次解析與報告產生工具")
    
    # 位置參數：支援輸入多個檔案或萬用字元
    parser.add_argument("input_files", nargs='+', help="輸入的 xml 檔案 (支援萬用字元，例如 trip*.xml)")
    
    # 選擇性參數：死鎖與車禍次數
    parser.add_argument("-d", "--deadlocks", type=int, default=0, help="該次模擬的死鎖次數 (預設: 0)")
    parser.add_argument("-c", "--collisions", type=int, default=0, help="該次模擬的車禍次數 (預設: 0)")

    args = parser.parse_args()

    # 展開所有檔案路徑 (解決 Windows CMD 不會自動展開 '*' 的問題)
    xml_files = []
    for pattern in args.input_files:
        matches = glob.glob(pattern)
        if matches:
            xml_files.extend(matches)
        else:
            # 如果找不到 match，還是加進去讓後續報錯提示
            xml_files.append(pattern)

    # 去除重複並排序
    xml_files = sorted(list(set(xml_files)))

    if not xml_files:
        print("⚠️ 找不到任何符合的 XML 檔案。")
        return

    print(f"🔍 找到 {len(xml_files)} 個 XML 檔案，開始批次分析...\n")
    
    success_count = 0
    for xml_file in xml_files:
        if analyze_tripinfo(xml_file, deadlocks=args.deadlocks, collisions=args.collisions):
            success_count += 1
            
    print(f"🎉 批次分析完成！共成功產生 {success_count}/{len(xml_files)} 份報告。")

if __name__ == "__main__":
    main()