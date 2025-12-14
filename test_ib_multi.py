#!/usr/bin/env python3
"""
測試不同的 IB API 握手格式
"""

import socket
import time

def test_v1_handshake(host, port):
    """舊版 API 握手 (v100 以下)"""
    print("\n--- 測試舊版握手 ---")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        
        # 舊版握手：直接發送客戶端版本號
        # 格式：版本號 + \0
        message = b"71\x001\x00"  # 版本 71, client_id 1
        sock.sendall(message)
        print(f"  已發送: {message}")
        
        response = sock.recv(1024)
        print(f"  ✅ 收到回應: {response[:100]}")
        sock.close()
        return True
    except socket.timeout:
        print("  ❌ 超時")
        sock.close()
        return False
    except Exception as e:
        print(f"  ❌ 錯誤: {e}")
        return False

def test_v2_handshake(host, port):
    """新版 API 握手 (v100+)"""
    print("\n--- 測試新版握手 (v100+) ---")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        
        # 新版握手
        # 1. 發送 "API\0"
        # 2. 發送版本字串 (長度前綴)
        api_marker = b"API\x00"
        version_str = b"v100..178"
        
        # 長度前綴 (4 bytes, big endian)
        import struct
        length = struct.pack("!I", len(version_str))
        
        message = api_marker + length + version_str
        sock.sendall(message)
        print(f"  已發送: {message}")
        
        response = sock.recv(1024)
        print(f"  ✅ 收到回應: {response[:100]}")
        sock.close()
        return True
    except socket.timeout:
        print("  ❌ 超時")
        sock.close()
        return False
    except Exception as e:
        print(f"  ❌ 錯誤: {e}")
        return False

def test_simple_connect(host, port):
    """只連接不發送"""
    print("\n--- 測試純連接（不發送資料）---")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((host, port))
        print("  ✅ 連接成功，等待 TWS 主動發送...")
        
        response = sock.recv(1024)
        print(f"  ✅ 收到回應: {response[:100]}")
        sock.close()
        return True
    except socket.timeout:
        print("  ❌ TWS 沒有主動發送資料")
        sock.close()
        return False
    except Exception as e:
        print(f"  ❌ 錯誤: {e}")
        return False

def test_ib_insync_direct(host, port):
    """直接用 ib_insync 測試"""
    print("\n--- 測試 ib_insync 直接連接 ---")
    try:
        # 設定 event loop
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except:
            pass
        
        from ib_insync import IB
        ib = IB()
        
        print(f"  嘗試連接到 {host}:{port}...")
        ib.connect(host, port, clientId=1, timeout=15)
        
        if ib.isConnected():
            print("  ✅ 連接成功！")
            accounts = ib.managedAccounts()
            print(f"  帳戶: {accounts}")
            ib.disconnect()
            return True
        else:
            print("  ❌ 連接失敗")
            return False
    except Exception as e:
        print(f"  ❌ 錯誤: {type(e).__name__}: {e}")
        return False

def main():
    host = "127.0.0.1"
    port = 7497
    
    print("=" * 50)
    print("IB API 連接測試（多種方法）")
    print("=" * 50)
    print(f"目標: {host}:{port}")
    
    # 測試各種方法
    test_simple_connect(host, port)
    test_v1_handshake(host, port)
    test_v2_handshake(host, port)
    test_ib_insync_direct(host, port)
    
    print("\n" + "=" * 50)
    print("測試完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
