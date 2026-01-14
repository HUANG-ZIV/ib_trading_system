#!/usr/bin/env python3
"""
測試 FeedHandler 和 EventBus
"""
import asyncio
import signal
import nest_asyncio
from datetime import datetime

nest_asyncio.apply()

# 設置 logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from ib_insync import IB, Forex
from core.event_bus import EventBus
from core.events import EventType, BarEvent
from data.feed_handler import FeedHandler, SubscriptionType

running = True
bar_count = 0

def on_bar_event(event: BarEvent):
    """EventBus 收到 BAR 事件"""
    global bar_count
    bar_count += 1
    print(f"[EventBus] Bar #{bar_count}: {event.symbol} Close={event.close:.5f}")

def signal_handler(signum, frame):
    global running
    print("\n收到停止信號...")
    running = False

async def main():
    global running, bar_count
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 1. 建立 EventBus
    print("1. 建立 EventBus...")
    event_bus = EventBus()
    event_bus.subscribe(EventType.BAR, on_bar_event)
    print("   已訂閱 BAR 事件")
    
    # 2. 連接 IB (直接使用 ib_insync)
    print("2. 連接 IB...")
    ib = IB()
    await ib.connectAsync('127.0.0.1', 7497, clientId=97)
    print(f"   連接成功！帳戶: {ib.managedAccounts()}")
    
    # 3. 建立 FeedHandler（手動設置 _ib）
    print("3. 建立 FeedHandler...")
    feed_handler = FeedHandler(event_bus=event_bus)
    feed_handler._ib = ib  # 手動設置
    
    # 4. 訂閱 USD/JPY
    print("4. 訂閱 USD/JPY...")
    contract = Forex('USDJPY')
    success = await feed_handler.subscribe(
        contract=contract,
        subscription_type=SubscriptionType.REALTIME_BAR,
        symbol_name="USD/JPY",
    )
    print(f"   訂閱結果: {success}")
    
    # 5. 等待數據
    print("5. 等待 Bar 數據...")
    print("-" * 50)
    
    seconds = 0
    while running:
        await asyncio.sleep(1)
        seconds += 1
        
        if seconds % 10 == 0:
            print(f"[狀態] 運行 {seconds}秒, 收到 {bar_count} 個 Bar")
    
    # 清理
    print("斷開連接...")
    ib.disconnect()
    print("完成")

if __name__ == "__main__":
    asyncio.run(main())
