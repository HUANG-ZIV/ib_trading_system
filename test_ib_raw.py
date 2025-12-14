#!/usr/bin/env python3
"""
原始 socket 測試 IB TWS 連接
"""

print("腳本開始執行...")

import socket
import struct
import time

def test_ib_handshake(host="127.0.0.1", port=7497):
    """測試 IB TWS API 握手"""
    print("=" * 50)
    print("IB TWS 原始連接測試")
    print("=" * 50)
    
    print(f"\n嘗試連接到 {host}:{port}...")
    
    try:
        # 建立 socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        # 連接
        sock.connect((host, port))
        print("✅ Socket 連接成功")
        
        # IB API 握手：發送版本號
        # 格式: "API\0" + 版本範圍
        # v100+ 使用新的協議
        
        # 發送 API 標識
        api_header = b"API\x00"
        
        # 發送支援的版本範圍 (v100 到 v178)
        version_str = "v100..178"
        version_msg = struct.pack(f"!I{len(version_str)}s", len(version_str), version_str.encode())
        
        message = api_header + version_msg
        sock.sendall(message)
        print("✅ 已發送 API 握手訊息")
        
        # 接收回應
        print("等待 TWS 回應...")
        
        try:
            response = sock.recv(1024)
            if response:
                print(f"✅ 收到回應: {len(response)} bytes")
                print(f"   原始資料: {response[:50]}...")
                print()
                print("=" * 50)
                print("✅ TWS API 連接測試成功！")
                print("=" * 50)
                print()
                print("TWS 設定正確，問題可能在 ib_insync 套件。")
                print("請嘗試更新 ib_insync：")
                print("  pip install --upgrade ib_insync")
            else:
                print("❌ 收到空回應")
        except socket.timeout:
            print("❌ 等待回應超時")
            print()
            print("這表示 TWS 收到連接但沒有回應。")
            print("請檢查：")
            print("  1. TWS 是否顯示任何連接提示或警告？")
            print("  2. 是否有彈出視窗要求確認 API 連接？")
        
        sock.close()
        
    except socket.timeout:
        print("❌ 連接超時")
    except ConnectionRefusedError:
        print("❌ 連接被拒絕 - TWS 可能未啟動或 API 未啟用")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

def check_multiple_ports():
    """檢查多個可能的端口"""
    print("\n" + "=" * 50)
    print("檢查常見 IB 端口")
    print("=" * 50)
    
    ports = [
        (7497, "TWS Paper Trading"),
        (7496, "TWS Live Trading"),
        (4002, "IB Gateway Paper Trading"),
        (4001, "IB Gateway Live Trading"),
    ]
    
    for port, description in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            
            if result == 0:
                print(f"  ✅ 端口 {port} ({description}) - 開放")
            else:
                print(f"  ❌ 端口 {port} ({description}) - 關閉")
        except:
            print(f"  ❌ 端口 {port} ({description}) - 無法測試")

if __name__ == "__main__":
    check_multiple_ports()
    print()
    test_ib_handshake()