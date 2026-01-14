"""
測試 XAUUSD 策略 - 簡化版
"""

from datetime import datetime
from ib_insync import IB, MarketOrder
import nest_asyncio
nest_asyncio.apply()

from config.symbols import create_commodity
from core.contracts import ContractFactory


def main():
    print("=" * 60)
    print("XAUUSD 測試策略")
    print("=" * 60)
    
    # 連接 IB
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=100)
    print(f"✅ 已連接 IB")
    
    # 建立合約
    factory = ContractFactory()
    xauusd_config = create_commodity("XAUUSD")
    xauusd_contract = factory.create(xauusd_config)
    ib.qualifyContracts(xauusd_contract)
    print(f"✅ XAUUSD 合約: {xauusd_contract}")
    
    # 訂閱即時數據
    print("\n📊 訂閱 5 秒 K 線...")
    bars = ib.reqRealTimeBars(xauusd_contract, 5, 'MIDPOINT', False)
    
    print("✅ 等待 K 線...\n")
    print("-" * 60)
    
    # 狀態
    bar_count = 0
    position = 0          # 0=無, 1=多, -1=空
    next_action = "BUY"
    bars_since_entry = 0
    
    TRIGGER_BARS = 3      # 每 3 根 K 線開倉
    CLOSE_BARS = 2        # 持倉 2 根 K 線後平倉
    QUANTITY = 1          # 1 盎司
    
    def on_bar(bars, hasNewBar):
        nonlocal bar_count, position, next_action, bars_since_entry
        
        if not hasNewBar:
            return
        
        bar = bars[-1]
        bar_count += 1
        
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Bar #{bar_count} | "
            f"Close: {bar.close:.2f} | "
            f"Position: {position}"
        )
        
        # 有持倉，檢查平倉
        if position != 0:
            bars_since_entry += 1
            
            if bars_since_entry >= CLOSE_BARS:
                if position == 1:
                    order = MarketOrder('SELL', QUANTITY)
                    print(f"  🔴 平倉 SELL {QUANTITY} @ {bar.close:.2f}")
                else:
                    order = MarketOrder('BUY', QUANTITY)
                    print(f"  🔴 平倉 BUY {QUANTITY} @ {bar.close:.2f}")
                
                trade = ib.placeOrder(xauusd_contract, order)
                position = 0
                bars_since_entry = 0
                return
        
        # 檢查開倉
        if bar_count % TRIGGER_BARS == 0 and position == 0:
            if next_action == "BUY":
                order = MarketOrder('BUY', QUANTITY)
                trade = ib.placeOrder(xauusd_contract, order)
                position = 1
                next_action = "SELL"
                print(f"  🔵 開倉 BUY {QUANTITY} @ {bar.close:.2f}")
            else:
                order = MarketOrder('SELL', QUANTITY)
                trade = ib.placeOrder(xauusd_contract, order)
                position = -1
                next_action = "BUY"
                print(f"  🔵 開倉 SELL {QUANTITY} @ {bar.close:.2f}")
            
            bars_since_entry = 0
    
    bars.updateEvent += on_bar
    
    print("按 Ctrl+C 停止\n")
    
    try:
        while True:
            ib.sleep(1)
    except KeyboardInterrupt:
        print("\n停止中...")
    
    ib.cancelRealTimeBars(bars)
    ib.disconnect()
    print("✅ 完成")


if __name__ == "__main__":
    main()
