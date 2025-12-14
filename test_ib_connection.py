#!/usr/bin/env python3
"""
簡單的 IB 連接測試腳本
"""

# Python 3.14+ 相容性修復（必須在最前面）
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 嘗試使用 nest_asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

import socket

def test_port(host, port):
    """測試端口是否開放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def main():
    print("=" * 50)
    print("IB 連接測試")
    print("=" * 50)
    
    host = "127.0.0.1"
    port = 7497  # Paper Trading 端口
    client_id = 1
    
    # 先測試端口是否開放
    print(f"\n1. 測試端口 {host}:{port} 是否開放...")
    if test_port(host, port):
        print(f"   ✅ 端口 {port} 開放")
    else:
        print(f"   ❌ 端口 {port} 未開放或無法連接")
        print()
        print("   可能的原因：")
        print("   - TWS 尚未完全啟動")
        print("   - API 端口設定不是 7497")
        print("   - 防火牆阻擋")
        print()
        print("   請嘗試：")
        print("   - 在 TWS 中確認 API 端口號碼")
        print("   - 重啟 TWS")
        return
    
    print(f"\n2. 嘗試連接 IB API...")
    
    from ib_insync import IB
    ib = IB()
    
    try:
        # 使用較長的超時時間
        ib.connect(host, port, clientId=client_id, timeout=20)
        
        if ib.isConnected():
            print("   ✅ 連接成功！")
            print()
            
            # 取得帳戶資訊
            print("3. 帳戶資訊：")
            for account in ib.managedAccounts():
                print(f"   帳戶: {account}")
            
            # 取得帳戶淨值
            account_values = ib.accountSummary()
            for av in account_values:
                if av.tag == "NetLiquidation":
                    print(f"   淨值: ${float(av.value):,.2f}")
                    break
            
            print()
            print("=" * 50)
            print("✅ IB 連接測試成功！")
            print("你可以運行 python run_live.py 來啟動交易系統")
            print("=" * 50)
            
            # 斷開連接
            ib.disconnect()
        else:
            print("   ❌ 連接失敗（未知原因）")
            
    except Exception as e:
        print(f"   ❌ 連接錯誤: {e}")
        print()
        print("   請確認 TWS 中的設定：")
        print("   - Edit → Global Configuration → API → Settings")
        print("   - ☑️ Enable ActiveX and Socket Clients")
        print("   - Socket port: 7497")
        print("   - ☑️ Allow connections from localhost only")

if __name__ == "__main__":
    main()