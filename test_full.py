#!/usr/bin/env python3
"""
完整測試 - 模擬 run_live.py 的流程
"""
import asyncio
import signal
import nest_asyncio
from datetime import datetime

nest_asyncio.apply()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

from ib_insync import IB, Forex
from core.event_bus import EventBus
from core.events import EventType, BarEvent, SignalEvent

running = True

def signal_handler(signum, frame):
    global running
    print("\n收到停止信號...")
    running = False

async def main():
    global running
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 1. 建立 EventBus
    print("1. 建立 EventBus...")
    event_bus = EventBus()
    
    # 訂閱 SIGNAL 事件來觀察策略輸出
    def on_signal(event):
        print(f"🚀 [SIGNAL] {event.symbol} {event.action} qty={event.suggested_quantity}")
    
    event_bus.subscribe(EventType.SIGNAL, on_signal)
    
    # 2. 連接 IB
    print("2. 連接 IB...")
    ib = IB()
    await ib.connectAsync('127.0.0.1', 7497, clientId=95)
    print(f"   連接成功！")
    
    # 3. 建立 StrategyEngine
    print("3. 建立 StrategyEngine...")
    from engine.strategy_engine import StrategyEngine
    strategy_engine = StrategyEngine(event_bus=event_bus)
    
    # 4. 建立策略
    print("4. 建立 TestStrategy...")
    from strategies.examples.test_strategy import TestStrategy
    strategy = TestStrategy(
        strategy_id="test",
        symbols=["USD/JPY"],
        trigger_bars=2,
        auto_close_bars=2,
        quantity=1,
    )
    strategy_engine.add_strategy(strategy)
    strategy.initialize()
    strategy.start()
    print(f"   策略已啟動，監控: {list(strategy.symbols)}")
    
    # 5. 啟動 StrategyEngine（訂閱 BAR 事件）
    print("5. 啟動 StrategyEngine...")
    strategy_engine.start()
    
    # 6. 訂閱數據並手動發布 BarEvent
    print("6. 訂閱 USD/JPY 數據...")
    contract = Forex('USDJPY')
    
    bars = ib.reqRealTimeBars(
        contract,
        barSize=5,
        whatToShow='MIDPOINT',
        useRTH=False,
    )
    
    bar_count = 0
    
    def on_bar_update(bars_data, has_new):
        nonlocal bar_count
        if has_new and bars_data:
            bar = bars_data[-1]
            bar_count += 1
            print(f"[IB] Bar #{bar_count}: Close={bar.close:.5f}")
            
            # 建立 BarEvent 並發布到 EventBus
            bar_event = BarEvent(
                event_type=EventType.BAR,
                symbol="USD/JPY",
                open=bar.open_,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=int(bar.volume) if bar.volume >= 0 else 0,
                bar_size="5 secs",
                bar_start=bar.time,
            )
            
            event_bus.publish(bar_event)
    
    bars.updateEvent += on_bar_update
    print("   已訂閱，等待數據...")
    print("-" * 50)
    
    # 7. 主循環
    seconds = 0
    while running and seconds < 60:  # 最多運行 60 秒
        await asyncio.sleep(1)
        seconds += 1
        
        if seconds % 10 == 0:
            print(f"[狀態] 運行 {seconds}秒, 收到 {bar_count} 個 Bar")
    
    # 清理
    print("斷開連接...")
    strategy_engine.stop()
    ib.disconnect()
    print("完成")

if __name__ == "__main__":
    asyncio.run(main())
