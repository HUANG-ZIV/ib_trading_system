"""
Data 模組 - 數據處理層

提供市場數據接收、聚合、儲存和快取功能：
- FeedHandler: 市場數據訂閱與接收
- BarAggregator: Tick 轉 Bar 聚合器
- Database: 數據持久化
- MarketDataCache: 內存快取
"""

from .feed_handler import (
    FeedHandler,
    SubscriptionInfo,
)

from .bar_aggregator import (
    BarAggregator,
    BarBuilder,
    BarSize,
)

from .database import (
    Database,
    get_database,
    TickData,
    BarData,
    TradeRecord,
)

from .cache import (
    MarketDataCache,
    get_market_data_cache,
)

__all__ = [
    # 數據接收
    "FeedHandler",
    "SubscriptionInfo",
    # Bar 聚合
    "BarAggregator",
    "BarBuilder",
    "BarSize",
    # 數據庫
    "Database",
    "get_database",
    "TickData",
    "BarData",
    "TradeRecord",
    # 快取
    "MarketDataCache",
    "get_market_data_cache",
]