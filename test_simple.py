#!/usr/bin/env python3
"""
簡單測試腳本 - 確認 IB 數據訂閱和策略執行
"""
import asyncio
import signal
import nest_asyncio
from datetime import datetime
from ib_insync import IB, Forex, Contract

nest_asyncio.apply()

# 全局變數
ib = None
running = True
bar_received = 0

def on_bar_update(bars, has_new):
    """收到 Bar 數據時的回調"""
    global bar_received
    if has_new and bars:
        bar = bars[-1]
        bar_received += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Bar #{bar_received}: Close={bar.close:.5f}")

def signal_handler(signum, frame):
    global running
    print("\n收到停止信號，關閉中...")
    running = False

async def main():
    global ib, running, bar_received
    
    # 設置信號處理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 連接 IB
    print("連接 IB...")
    ib = IB()
    await ib.connectAsync('127.0.0.1', 7497, clientId=99)
    print(f"連接成功！帳戶: {ib.managedAccounts()}")
    
    # 建立合約 - 使用 USD/JPY
    contract = Forex('USDJPY')
    print(f"訂閱合約: {contract.symbol}{contract.currency}")
    
    # 訂閱即時 Bar（5 秒）
    bars = ib.reqRealTimeBars(
        contract,
        barSize=5,
        whatToShow='MIDPOINT',
        useRTH=False,
    )
    
    # 註冊回調
    bars.updateEvent += on_bar_update
    print("已註冊 Bar 回調，等待數據...")
    print("按 Ctrl+C 停止")
    print("-" * 50)
    
    # 等待數據
    seconds = 0
    while running:
        await asyncio.sleep(1)
        seconds += 1
        
        # 每 10 秒輸出狀態
        if seconds % 10 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"運行 {seconds}秒, 收到 {bar_received} 個 Bar")
    
    # 清理
    print("斷開連接...")
    ib.disconnect()
    print("完成")

if __name__ == "__main__":
    asyncio.run(main())
