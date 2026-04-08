import xml.etree.ElementTree as ET
import os
import re
import argparse
import glob
from datetime import datetime


def extract_stats_from_log(log_filepath):
    """
    從 RL/GA/Baseline 的 log txt 自動提取死鎖與車禍次數。
    回傳 (collisions, deadlocks)
    """
    if not os.path.exists(log_filepath):
        print(f"[警告] 找不到 log 檔案: {log_filepath}")
        return 0, 0

    with open(log_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    collisions = len(re.findall(r"REAL_COLLISION", content))
    deadlocks = len(re.findall(r"死鎖救援|死鎖移除|瞬移失敗", content))

    print(f"📋 [Log 解析] {os.path.basename(log_filepath)}")
    print(f"   → 車禍: {collisions} 次 | 死鎖處理: {deadlocks} 次")
    return collisions, deadlocks


def find_xml_from_log(log_filepath):
    """
    嘗試從 log 檔案內容找到對應的 tripinfo XML 檔名。
    """
    with open(log_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 抓 tripinfo XML 檔名（各種格式都找）
    matches = re.findall(r"tripinfo[\w_\-.]+\.xml", content)
    if matches:
        # 取最後一個（通常是最終產出的那個）
        return matches[-1]
    return None


def analyze_tripinfo(xml_filepath, deadlocks=0, collisions=0):
    """
    解析單一 SUMO 的 tripinfo.xml 並自動產出對應檔名的 TXT 報告。
    """
    if not os.path.exists(xml_filepath):
        print(f"[警告] 找不到指定的 XML 檔案: {xml_filepath}，已跳過。")
        return False

    base_name = os.path.splitext(xml_filepath)[0]
    output_txt_filepath = f"{base_name}_analysis.txt"

    total_vehicles = 0
    total_time_loss = 0.0
    total_waiting_time = 0.0
    total_duration = 0.0

    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        for trip in root.findall('tripinfo'):
            total_vehicles += 1
            total_time_loss += float(trip.get('timeLoss', 0))
            total_waiting_time += float(trip.get('waitingTime', 0))
            total_duration += float(trip.get('duration', 0))

        if total_vehicles > 0:
            avg_time_loss = total_time_loss / total_vehicles
            avg_waiting_time = total_waiting_time / total_vehicles
            avg_duration = total_duration / total_vehicles
        else:
            avg_time_loss = avg_waiting_time = avg_duration = 0.0

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
            f" 🔒 總死鎖處理次數 (Deadlocks)   : {deadlocks} 次",
            "",
            "【2. 關鍵時間指標 (所有車輛平均)】",
            f" ⏱️ 實際平均延遲 (Avg TimeLoss)   : {avg_time_loss:.2f} 秒",
            f" 🛑 絕對靜止時間 (Avg WaitingTime): {avg_waiting_time:.2f} 秒",
            f" 🛣️ 總旅行時間   (Avg Duration)   : {avg_duration:.2f} 秒",
            "=================================================="
        ]

        with open(output_txt_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"✅ 成功: {xml_filepath} -> {output_txt_filepath}")
        print(f"   [表現] 延遲: {avg_time_loss:.2f}s | 車禍: {collisions} | 死鎖: {deadlocks}\n")
        return True

    except Exception as e:
        print(f"[錯誤] 解析 {xml_filepath} 時發生問題: {str(e)}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SUMO tripinfo.xml 批次解析與報告產生工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
使用範例：

  【模式 1】直接指定 XML（手動輸入死鎖/車禍數）：
    python3 tripinfo_analyzer.py tripinfo_BASELINE_*.xml
    python3 tripinfo_analyzer.py tripinfo_GATEST_*.xml -c 74 -d 139
    python3 tripinfo_analyzer.py tripinfo_RL_*.xml -c 66 -d 91

  【模式 2】從 log 自動提取死鎖/車禍，並自動對應 XML：
    python3 tripinfo_analyzer.py --from-log execute_BASELINE_*.txt
    python3 tripinfo_analyzer.py --from-log execute_GATEST_*.txt
    python3 tripinfo_analyzer.py --from-log execute_RL_*_test.txt

  【模式 3】指定 XML 但從 log 自動提取死鎖/車禍數：
    python3 tripinfo_analyzer.py tripinfo_GATEST_*.xml --from-log execute_GATEST_*.txt
        """
    )

    parser.add_argument("input_files", nargs='*',
                        help="輸入的 xml 檔案（支援萬用字元）。\n"
                             "若使用 --from-log 且不指定 XML，會自動從 log 內找 XML 檔名。")
    parser.add_argument("-d", "--deadlocks", type=int, default=None,
                        help="手動指定死鎖次數（不使用 --from-log 時）")
    parser.add_argument("-c", "--collisions", type=int, default=None,
                        help="手動指定車禍次數（不使用 --from-log 時）")
    parser.add_argument("--from-log", nargs='+', metavar="LOG_FILE",
                        help="從指定的 log txt 自動提取車禍與死鎖次數\n"
                             "（若未指定 XML，也會嘗試從 log 內容自動找對應 XML）")

    args = parser.parse_args()

    # ==========================================
    # 整理 XML 檔案清單（只接受 .xml）
    # ==========================================
    xml_files = []
    for pattern in (args.input_files or []):
        matches = glob.glob(pattern)
        found = matches if matches else [pattern]
        for f in found:
            if f.endswith('.xml'):
                xml_files.append(f)
            else:
                print(f"⚠️ 跳過非 XML 檔案: {f}（請用 --from-log 傳入 log）")
    xml_files = sorted(set(xml_files))

    # ==========================================
    # 整理 log 檔案清單，提取死鎖/車禍（只接受 .txt）
    # ==========================================
    log_collisions = 0
    log_deadlocks = 0

    if args.from_log:
        log_files = []
        for pattern in args.from_log:
            matches = glob.glob(pattern)
            found = matches if matches else [pattern]
            for f in found:
                if f.endswith('.txt'):
                    log_files.append(f)
                elif f.endswith('.xml'):
                    # 使用者不小心把 XML 傳進 --from-log，自動移到 xml_files
                    print(f"⚠️ {os.path.basename(f)} 是 XML，已自動移至分析清單")
                    xml_files.append(f)
                else:
                    print(f"⚠️ 跳過未知格式: {f}")
        log_files = sorted(set(log_files))

        for log_file in log_files:
            c, d = extract_stats_from_log(log_file)
            log_collisions += c
            log_deadlocks += d

            # 如果沒有指定 XML，嘗試從 log 內容找
            if not xml_files:
                xml_name = find_xml_from_log(log_file)
                if xml_name and os.path.exists(xml_name):
                    xml_files.append(xml_name)
                    print(f"   → 自動找到 XML: {xml_name}")
                elif xml_name:
                    print(f"   ⚠️ log 中提到 {xml_name}，但找不到該檔案")

    # ==========================================
    # 決定最終使用的死鎖/車禍數
    # ==========================================
    final_collisions = args.collisions if args.collisions is not None else log_collisions
    final_deadlocks = args.deadlocks if args.deadlocks is not None else log_deadlocks

    if not xml_files:
        print("⚠️ 找不到任何 XML 檔案可以分析。")
        print("   提示：請確認 XML 檔案存在，或用 --from-log 搭配有記錄 XML 檔名的 log")
        return

    xml_files = sorted(set(xml_files))
    print(f"\n🔍 找到 {len(xml_files)} 個 XML 檔案，開始分析...\n")

    success_count = 0
    for xml_file in xml_files:
        if analyze_tripinfo(xml_file,
                            deadlocks=final_deadlocks,
                            collisions=final_collisions):
            success_count += 1

    print(f"🎉 完成！共成功產生 {success_count}/{len(xml_files)} 份報告。")


if __name__ == "__main__":
    main()