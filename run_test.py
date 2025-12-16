#!/usr/bin/env python3
"""
run_test.py - 異常處理測試
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ib_insync import IB, Order, util
from core.contracts import ContractFactory

util.startLoop()

# 設定
SYMBOLS = ["XAUUSD", "USD/JPY"]
TRIGGER_BARS = 3
CLOSE_BARS = 2
QUANTITY = {"XAUUSD": 1, "USD/JPY": 10000}

# 狀態
ib = None
contracts = {}
bar_count = {}
position = {}
next_action = {}
bars_since_entry = {}
trade_history = []
bars_subscriptions = []


def create_market_order(action, qty):
    order = Order()
    order.action = action
    order.totalQuantity = qty
    order.orderType = 'MKT'
    order.tif = 'GTC'
    return order


def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")


def on_order_status(trade, symbol):
    status = trade.orderStatus.status
    action = trade.order.action
    qty = trade.order.totalQuantity
    
    if status == 'Filled':
        fill_price = trade.orderStatus.avgFillPrice
        log(f"     ✅ 成交: {symbol} {action} {qty} @ {fill_price:.4f}")
        trade_history.append({
            'time': datetime.now(),
            'symbol': symbol,
            'action': action,
            'qty': qty,
            'price': fill_price,
        })
    elif status == 'Cancelled':
        log(f"     ❌ 取消: {symbol} {action} {qty}")
    elif status == 'Submitted':
        log(f"     📤 已提交: {symbol} {action} {qty}")


def on_bar_update(bars, hasNewBar, symbol):
    global bar_count, position, next_action, bars_since_entry
    
    if not hasNewBar:
        return
    
    bar = bars[-1]
    bar_count[symbol] += 1
    
    log(
        f"[{symbol}] Bar #{bar_count[symbol]} | "
        f"Close: {bar.close:.4f} | "
        f"Pos: {position[symbol]} | "
        f"Next: {next_action[symbol]}"
    )
    
    contract = contracts[symbol]
    qty = QUANTITY.get(symbol, 1)
    
    if position[symbol] != 0:
        bars_since_entry[symbol] += 1
        
        if bars_since_entry[symbol] >= CLOSE_BARS:
            if position[symbol] == 1:
                order = create_market_order('SELL', qty)
                log(f"  🔴 平倉 SELL {qty}")
            else:
                order = create_market_order('BUY', qty)
                log(f"  🔴 平倉 BUY {qty}")
            
            trade = ib.placeOrder(contract, order)
            trade.statusEvent += lambda t, s=symbol: on_order_status(t, s)
            position[symbol] = 0
            bars_since_entry[symbol] = 0
            return
    
    if bar_count[symbol] % TRIGGER_BARS == 0 and position[symbol] == 0:
        action = next_action[symbol]
        
        if action == "BUY":
            order = create_market_order('BUY', qty)
            trade = ib.placeOrder(contract, order)
            trade.statusEvent += lambda t, s=symbol: on_order_status(t, s)
            log(f"  🔵 開倉 BUY {qty}")
            position[symbol] = 1
            next_action[symbol] = "SELL"
        else:
            order = create_market_order('SELL', qty)
            trade = ib.placeOrder(contract, order)
            trade.statusEvent += lambda t, s=symbol: on_order_status(t, s)
            log(f"  🔵 開倉 SELL {qty}")
            position[symbol] = -1
            next_action[symbol] = "BUY"
        
        bars_since_entry[symbol] = 0


def close_all_positions():
    """平掉所有持倉"""
    log("📤 檢查並平倉...")
    
    for symbol in SYMBOLS:
        if position[symbol] != 0:
            contract = contracts[symbol]
            qty = QUANTITY.get(symbol, 1)
            
            if position[symbol] == 1:
                order = create_market_order('SELL', qty)
                log(f"  🔴 緊急平倉 {symbol} SELL {qty}")
            else:
                order = create_market_order('BUY', qty)
                log(f"  �� 緊急平倉 {symbol} BUY {qty}")
            
            trade = ib.placeOrder(contract, order)
            trade.statusEvent += lambda t, s=symbol: on_order_status(t, s)
            position[symbol] = 0
    
    # 等待訂單處理
    ib.sleep(2)


def cancel_all_orders():
    """取消所有未成交訂單"""
    open_trades = ib.openTrades()
    if open_trades:
        log(f"�� 取消 {len(open_trades)} 個未成交訂單...")
        for trade in open_trades:
            ib.cancelOrder(trade.order)
        ib.sleep(1)
    else:
        log("✅ 無未成交訂單")


def print_summary():
    print("\n" + "=" * 60)
    print("交易摘要")
    print("=" * 60)
    print(f"總交易次數: {len(trade_history)}")
    
    for symbol in SYMBOLS:
        symbol_trades = [t for t in trade_history if t['symbol'] == symbol]
        print(f"\n📊 {symbol}:")
        print(f"  交易次數: {len(symbol_trades)}")
        print(f"  買入: {len([t for t in symbol_trades if t['action'] == 'BUY'])} 筆")
        print(f"  賣出: {len([t for t in symbol_trades if t['action'] == 'SELL'])} 筆")
        for t in symbol_trades:
            print(f"    {t['time'].strftime('%H:%M:%S')} {t['action']:4} {t['qty']:>8} @ {t['price']:.4f}")
    
    print("\n" + "=" * 60)
    
    # 顯示最終持倉狀態
    print("最終持倉狀態:")
    for symbol in SYMBOLS:
        status = "無持倉" if position[symbol] == 0 else f"{'多' if position[symbol] > 0 else '空'} {abs(position[symbol])}"
        print(f"  {symbol}: {status}")
    print("=" * 60)


def shutdown():
    """安全關閉"""
    log("=" * 50)
    log("🛑 開始安全關閉程序...")
    log("=" * 50)
    
    # 1. 取消未成交訂單
    cancel_all_orders()
    
    # 2. 平掉所有持倉
    close_all_positions()
    
    # 3. 取消數據訂閱
    log("📤 取消數據訂閱...")
    for bars in bars_subscriptions:
        try:
            ib.cancelRealTimeBars(bars)
        except:
            pass
    
    # 4. 印出摘要
    print_summary()
    
    # 5. 斷開連線
    ib.disconnect()
    log("✅ 已安全斷開連接")


# 主程式
print("=" * 60)
print("異常處理測試")
print(f"觸發間隔: {TRIGGER_BARS} 根 K 線 (約 {TRIGGER_BARS * 5} 秒)")
print(f"平倉間隔: {CLOSE_BARS} 根 K 線 (約 {CLOSE_BARS * 5} 秒)")
print("=" * 60)

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=100)
log("✅ 已連接 IB")

for av in ib.accountSummary():
    if av.tag == "NetLiquidation":
        log(f"💰 帳戶淨值: ${float(av.value):,.2f}")
        break

factory = ContractFactory()

for symbol in SYMBOLS:
    bar_count[symbol] = 0
    position[symbol] = 0
    next_action[symbol] = "BUY"
    bars_since_entry[symbol] = 0
    
    if symbol == "XAUUSD":
        contract = factory.commodity(symbol)
    elif "/" in symbol:
        parts = symbol.split("/")
        contract = factory.forex(parts[0], parts[1])
    else:
        contract = factory.stock(symbol)
    
    ib.qualifyContracts(contract)
    contracts[symbol] = contract
    log(f"✅ {symbol}")
    
    bars = ib.reqRealTimeBars(contract, 5, 'MIDPOINT', False)
    bars.updateEvent += lambda b, h, s=symbol: on_bar_update(b, h, s)
    bars_subscriptions.append(bars)

print("")
log("📊 等待 K 線...")
log("💡 測試: 在有持倉時按 Ctrl+C，觀察是否自動平倉")
log("按 Ctrl+C 停止")
print("-" * 60)

try:
    ib.run()
except KeyboardInterrupt:
    print("\n")
    log("⚠️ 收到 Ctrl+C 信號")
    shutdown()
