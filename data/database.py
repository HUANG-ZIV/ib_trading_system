"""
Database 模組 - 數據持久化

使用 SQLAlchemy 管理市場數據和交易記錄的存儲
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
import threading

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
    Index,
    text,
    and_,
    or_,
    desc,
    asc,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Session,
    scoped_session,
)
from sqlalchemy.pool import QueuePool

from core.events import TickEvent, BarEvent, FillEvent, OrderAction


# 設定 logger
logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()


# ============================================================
# ORM 模型
# ============================================================

class TickData(Base):
    """Tick 數據表"""
    
    __tablename__ = "tick_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # 價格
    bid = Column(Float)
    ask = Column(Float)
    last = Column(Float)
    
    # 數量
    bid_size = Column(Integer)
    ask_size = Column(Integer)
    last_size = Column(Integer)
    volume = Column(Integer)
    
    # OHLC（當日）
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    
    # 建立複合索引
    __table_args__ = (
        Index("idx_tick_symbol_timestamp", "symbol", "timestamp"),
    )
    
    def to_event(self) -> TickEvent:
        """轉換為 TickEvent"""
        return TickEvent(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bid=self.bid,
            ask=self.ask,
            last=self.last,
            bid_size=self.bid_size,
            ask_size=self.ask_size,
            last_size=self.last_size,
            volume=self.volume,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
        )
    
    @classmethod
    def from_event(cls, event: TickEvent) -> "TickData":
        """從 TickEvent 建立"""
        return cls(
            symbol=event.symbol,
            timestamp=event.timestamp,
            bid=event.bid,
            ask=event.ask,
            last=event.last,
            bid_size=event.bid_size,
            ask_size=event.ask_size,
            last_size=event.last_size,
            volume=event.volume,
            open=event.open,
            high=event.high,
            low=event.low,
            close=event.close,
        )


class BarData(Base):
    """K 線數據表"""
    
    __tablename__ = "bar_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    bar_size = Column(String(20), nullable=False, index=True)
    bar_start = Column(DateTime, nullable=False, index=True)
    bar_end = Column(DateTime)
    
    # OHLCV
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, default=0)
    
    # 額外資訊
    vwap = Column(Float)
    trade_count = Column(Integer)
    
    # 時間戳
    created_at = Column(DateTime, default=datetime.now)
    
    # 複合索引
    __table_args__ = (
        Index("idx_bar_symbol_size_start", "symbol", "bar_size", "bar_start"),
    )
    
    def to_event(self) -> BarEvent:
        """轉換為 BarEvent"""
        return BarEvent(
            symbol=self.symbol,
            bar_size=self.bar_size,
            bar_start=self.bar_start,
            bar_end=self.bar_end,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            vwap=self.vwap,
            trade_count=self.trade_count,
        )
    
    @classmethod
    def from_event(cls, event: BarEvent) -> "BarData":
        """從 BarEvent 建立"""
        return cls(
            symbol=event.symbol,
            bar_size=event.bar_size,
            bar_start=event.bar_start,
            bar_end=event.bar_end,
            open=event.open,
            high=event.high,
            low=event.low,
            close=event.close,
            volume=event.volume,
            vwap=event.vwap,
            trade_count=event.trade_count,
        )


class TradeRecord(Base):
    """交易記錄表"""
    
    __tablename__ = "trade_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 訂單識別
    order_id = Column(Integer, index=True)
    execution_id = Column(String(50), unique=True)
    perm_id = Column(Integer)
    
    # 交易內容
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # BUY/SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    
    # 成本
    commission = Column(Float, default=0.0)
    
    # 時間
    execution_time = Column(DateTime, nullable=False, index=True)
    
    # 來源
    strategy_id = Column(String(50), index=True)
    
    # 盈虧（平倉時計算）
    realized_pnl = Column(Float)
    
    # 額外資訊
    exchange = Column(String(20))
    notes = Column(String(500))
    
    # 時間戳
    created_at = Column(DateTime, default=datetime.now)
    
    # 索引
    __table_args__ = (
        Index("idx_trade_symbol_time", "symbol", "execution_time"),
        Index("idx_trade_strategy_time", "strategy_id", "execution_time"),
    )
    
    @property
    def value(self) -> float:
        """交易金額"""
        return self.quantity * self.price
    
    @property
    def net_value(self) -> float:
        """淨金額（扣除手續費）"""
        if self.action == "BUY":
            return -(self.value + self.commission)
        else:
            return self.value - self.commission
    
    @classmethod
    def from_fill_event(cls, event: FillEvent) -> "TradeRecord":
        """從 FillEvent 建立"""
        return cls(
            order_id=event.order_id,
            execution_id=event.execution_id,
            symbol=event.symbol,
            action=event.action.value if isinstance(event.action, OrderAction) else event.action,
            quantity=event.quantity,
            price=event.price,
            commission=event.commission,
            execution_time=event.execution_time or event.timestamp,
            strategy_id=event.strategy_id,
            exchange=event.exchange,
        )


class DailyPnL(Base):
    """每日盈虧表"""
    
    __tablename__ = "daily_pnl"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)
    
    # 盈虧
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    
    # 交易統計
    trade_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    
    # 金額統計
    total_commission = Column(Float, default=0.0)
    gross_profit = Column(Float, default=0.0)
    gross_loss = Column(Float, default=0.0)
    
    # 帳戶
    account_value = Column(Float)
    
    # 時間戳
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ============================================================
# Database 類
# ============================================================

class Database:
    """
    數據庫管理器
    
    管理資料庫連接、session 和 CRUD 操作
    
    使用方式:
        db = Database("sqlite:///data.db")
        
        # 儲存 Bar 數據
        db.save_bar(bar_event)
        
        # 查詢歷史數據
        bars = db.get_bars("AAPL", "1min", days=7)
        
        # 使用 session context
        with db.session_scope() as session:
            session.query(BarData).filter_by(symbol="AAPL").all()
    """
    
    def __init__(
        self,
        connection_string: str = "sqlite:///data_store/trading.db",
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """
        初始化數據庫
        
        Args:
            connection_string: 資料庫連接字串
            echo: 是否輸出 SQL 語句
            pool_size: 連接池大小
            max_overflow: 最大溢出連接數
        """
        self._connection_string = connection_string
        
        # 確保目錄存在
        if connection_string.startswith("sqlite:///"):
            db_path = connection_string.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 建立引擎
        if connection_string.startswith("sqlite"):
            # SQLite 不支援連接池參數
            self._engine = create_engine(
                connection_string,
                echo=echo,
                connect_args={"check_same_thread": False},
            )
        else:
            self._engine = create_engine(
                connection_string,
                echo=echo,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )
        
        # 建立 session 工廠
        self._session_factory = sessionmaker(bind=self._engine)
        self._scoped_session = scoped_session(self._session_factory)
        
        # 建立所有表
        Base.metadata.create_all(self._engine)
        
        logger.info(f"數據庫初始化完成: {connection_string}")
    
    # ========== Session 管理 ==========
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Session context manager
        
        自動處理 commit/rollback
        
        使用方式:
            with db.session_scope() as session:
                session.add(record)
        """
        session = self._scoped_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"數據庫錯誤: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """取得 session（需手動管理）"""
        return self._scoped_session()
    
    # ========== Tick 數據操作 ==========
    
    def save_tick(self, event: TickEvent) -> bool:
        """
        儲存 Tick 數據
        
        Args:
            event: TickEvent
            
        Returns:
            是否成功
        """
        try:
            with self.session_scope() as session:
                tick = TickData.from_event(event)
                session.add(tick)
            return True
        except Exception as e:
            logger.error(f"儲存 Tick 失敗: {e}")
            return False
    
    def save_ticks(self, events: List[TickEvent]) -> int:
        """
        批量儲存 Tick 數據
        
        Args:
            events: TickEvent 列表
            
        Returns:
            成功儲存的數量
        """
        try:
            with self.session_scope() as session:
                ticks = [TickData.from_event(e) for e in events]
                session.add_all(ticks)
            return len(events)
        except Exception as e:
            logger.error(f"批量儲存 Tick 失敗: {e}")
            return 0
    
    def get_ticks(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[TickEvent]:
        """
        查詢 Tick 數據
        
        Args:
            symbol: 標的代碼
            start_time: 開始時間
            end_time: 結束時間
            limit: 最大數量
            
        Returns:
            TickEvent 列表
        """
        with self.session_scope() as session:
            query = session.query(TickData).filter(TickData.symbol == symbol)
            
            if start_time:
                query = query.filter(TickData.timestamp >= start_time)
            if end_time:
                query = query.filter(TickData.timestamp <= end_time)
            
            query = query.order_by(asc(TickData.timestamp)).limit(limit)
            
            return [tick.to_event() for tick in query.all()]
    
    # ========== Bar 數據操作 ==========
    
    def save_bar(self, event: BarEvent) -> bool:
        """
        儲存 Bar 數據
        
        Args:
            event: BarEvent
            
        Returns:
            是否成功
        """
        try:
            with self.session_scope() as session:
                bar = BarData.from_event(event)
                session.add(bar)
            return True
        except Exception as e:
            logger.error(f"儲存 Bar 失敗: {e}")
            return False
    
    def save_bars(self, events: List[BarEvent]) -> int:
        """
        批量儲存 Bar 數據
        
        Args:
            events: BarEvent 列表
            
        Returns:
            成功儲存的數量
        """
        try:
            with self.session_scope() as session:
                bars = [BarData.from_event(e) for e in events]
                session.add_all(bars)
            return len(events)
        except Exception as e:
            logger.error(f"批量儲存 Bar 失敗: {e}")
            return 0
    
    def get_bars(
        self,
        symbol: str,
        bar_size: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: Optional[int] = None,
        limit: int = 10000,
    ) -> List[BarEvent]:
        """
        查詢 Bar 數據
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            start_time: 開始時間
            end_time: 結束時間
            days: 最近幾天（與 start_time 二選一）
            limit: 最大數量
            
        Returns:
            BarEvent 列表
        """
        with self.session_scope() as session:
            query = session.query(BarData).filter(
                BarData.symbol == symbol,
                BarData.bar_size == bar_size,
            )
            
            if days:
                start_time = datetime.now() - timedelta(days=days)
            
            if start_time:
                query = query.filter(BarData.bar_start >= start_time)
            if end_time:
                query = query.filter(BarData.bar_start <= end_time)
            
            query = query.order_by(asc(BarData.bar_start)).limit(limit)
            
            return [bar.to_event() for bar in query.all()]
    
    def get_latest_bar(
        self,
        symbol: str,
        bar_size: str,
    ) -> Optional[BarEvent]:
        """
        取得最新的 Bar
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            
        Returns:
            BarEvent 或 None
        """
        with self.session_scope() as session:
            bar = session.query(BarData).filter(
                BarData.symbol == symbol,
                BarData.bar_size == bar_size,
            ).order_by(desc(BarData.bar_start)).first()
            
            return bar.to_event() if bar else None
    
    # ========== 交易記錄操作 ==========
    
    def save_trade(self, event: FillEvent) -> bool:
        """
        儲存交易記錄
        
        Args:
            event: FillEvent
            
        Returns:
            是否成功
        """
        try:
            with self.session_scope() as session:
                trade = TradeRecord.from_fill_event(event)
                session.add(trade)
            return True
        except Exception as e:
            logger.error(f"儲存交易記錄失敗: {e}")
            return False
    
    def save_trade_record(self, record: TradeRecord) -> bool:
        """
        直接儲存 TradeRecord
        
        Args:
            record: TradeRecord 物件
            
        Returns:
            是否成功
        """
        try:
            with self.session_scope() as session:
                session.add(record)
            return True
        except Exception as e:
            logger.error(f"儲存交易記錄失敗: {e}")
            return False
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: Optional[int] = None,
        limit: int = 1000,
    ) -> List[TradeRecord]:
        """
        查詢交易記錄
        
        Args:
            symbol: 標的代碼（可選）
            strategy_id: 策略 ID（可選）
            start_time: 開始時間
            end_time: 結束時間
            days: 最近幾天
            limit: 最大數量
            
        Returns:
            TradeRecord 列表
        """
        with self.session_scope() as session:
            query = session.query(TradeRecord)
            
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
            if strategy_id:
                query = query.filter(TradeRecord.strategy_id == strategy_id)
            
            if days:
                start_time = datetime.now() - timedelta(days=days)
            
            if start_time:
                query = query.filter(TradeRecord.execution_time >= start_time)
            if end_time:
                query = query.filter(TradeRecord.execution_time <= end_time)
            
            query = query.order_by(desc(TradeRecord.execution_time)).limit(limit)
            
            # 返回 detached 物件
            return [self._detach_record(t) for t in query.all()]
    
    def _detach_record(self, record: TradeRecord) -> TradeRecord:
        """建立 detached 副本"""
        return TradeRecord(
            id=record.id,
            order_id=record.order_id,
            execution_id=record.execution_id,
            perm_id=record.perm_id,
            symbol=record.symbol,
            action=record.action,
            quantity=record.quantity,
            price=record.price,
            commission=record.commission,
            execution_time=record.execution_time,
            strategy_id=record.strategy_id,
            realized_pnl=record.realized_pnl,
            exchange=record.exchange,
            notes=record.notes,
            created_at=record.created_at,
        )
    
    def get_trade_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        取得交易統計摘要
        
        Args:
            start_time: 開始時間
            end_time: 結束時間
            days: 最近幾天
            
        Returns:
            統計摘要字典
        """
        trades = self.get_trades(
            start_time=start_time,
            end_time=end_time,
            days=days,
            limit=10000,
        )
        
        if not trades:
            return {
                "trade_count": 0,
                "total_commission": 0.0,
                "total_pnl": 0.0,
            }
        
        total_commission = sum(t.commission for t in trades)
        total_pnl = sum(t.realized_pnl or 0 for t in trades)
        
        buys = [t for t in trades if t.action == "BUY"]
        sells = [t for t in trades if t.action == "SELL"]
        
        return {
            "trade_count": len(trades),
            "buy_count": len(buys),
            "sell_count": len(sells),
            "total_commission": total_commission,
            "total_pnl": total_pnl,
            "net_pnl": total_pnl - total_commission,
            "symbols": list(set(t.symbol for t in trades)),
        }
    
    # ========== 每日盈虧操作 ==========
    
    def save_daily_pnl(
        self,
        date: datetime,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        trade_count: int = 0,
        total_commission: float = 0.0,
    ) -> bool:
        """
        儲存每日盈虧
        
        Args:
            date: 日期
            realized_pnl: 已實現盈虧
            unrealized_pnl: 未實現盈虧
            trade_count: 交易次數
            total_commission: 總手續費
            
        Returns:
            是否成功
        """
        try:
            with self.session_scope() as session:
                # 檢查是否已存在
                existing = session.query(DailyPnL).filter(
                    DailyPnL.date == date.replace(hour=0, minute=0, second=0, microsecond=0)
                ).first()
                
                if existing:
                    existing.realized_pnl = realized_pnl
                    existing.unrealized_pnl = unrealized_pnl
                    existing.total_pnl = realized_pnl + unrealized_pnl
                    existing.trade_count = trade_count
                    existing.total_commission = total_commission
                else:
                    pnl = DailyPnL(
                        date=date.replace(hour=0, minute=0, second=0, microsecond=0),
                        realized_pnl=realized_pnl,
                        unrealized_pnl=unrealized_pnl,
                        total_pnl=realized_pnl + unrealized_pnl,
                        trade_count=trade_count,
                        total_commission=total_commission,
                    )
                    session.add(pnl)
            
            return True
        except Exception as e:
            logger.error(f"儲存每日盈虧失敗: {e}")
            return False
    
    def get_daily_pnl(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: Optional[int] = None,
    ) -> List[DailyPnL]:
        """
        查詢每日盈虧
        
        Args:
            start_date: 開始日期
            end_date: 結束日期
            days: 最近幾天
            
        Returns:
            DailyPnL 列表
        """
        with self.session_scope() as session:
            query = session.query(DailyPnL)
            
            if days:
                start_date = datetime.now() - timedelta(days=days)
            
            if start_date:
                query = query.filter(DailyPnL.date >= start_date)
            if end_date:
                query = query.filter(DailyPnL.date <= end_date)
            
            query = query.order_by(asc(DailyPnL.date))
            
            return query.all()
    
    # ========== 工具方法 ==========
    
    def cleanup_old_data(
        self,
        days: int = 90,
        cleanup_ticks: bool = True,
        cleanup_bars: bool = False,
    ) -> Dict[str, int]:
        """
        清理舊數據
        
        Args:
            days: 保留最近幾天
            cleanup_ticks: 是否清理 Tick
            cleanup_bars: 是否清理 Bar
            
        Returns:
            {table: deleted_count}
        """
        cutoff = datetime.now() - timedelta(days=days)
        result = {}
        
        with self.session_scope() as session:
            if cleanup_ticks:
                deleted = session.query(TickData).filter(
                    TickData.timestamp < cutoff
                ).delete()
                result["ticks"] = deleted
            
            if cleanup_bars:
                deleted = session.query(BarData).filter(
                    BarData.bar_start < cutoff
                ).delete()
                result["bars"] = deleted
        
        logger.info(f"清理舊數據: {result}")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """取得數據庫統計"""
        with self.session_scope() as session:
            return {
                "tick_count": session.query(TickData).count(),
                "bar_count": session.query(BarData).count(),
                "trade_count": session.query(TradeRecord).count(),
                "daily_pnl_count": session.query(DailyPnL).count(),
            }
    
    def close(self) -> None:
        """關閉數據庫連接"""
        self._scoped_session.remove()
        self._engine.dispose()
        logger.info("數據庫連接已關閉")


# ============================================================
# 全局單例
# ============================================================

_database: Optional[Database] = None
_database_lock = threading.Lock()


def get_database(
    connection_string: str = "sqlite:///data_store/trading.db",
    **kwargs,
) -> Database:
    """
    取得全局 Database 實例（單例模式）
    
    Args:
        connection_string: 資料庫連接字串
        **kwargs: 傳遞給 Database 的其他參數
        
    Returns:
        Database 實例
    """
    global _database
    
    if _database is None:
        with _database_lock:
            if _database is None:
                _database = Database(connection_string, **kwargs)
    
    return _database


def reset_database() -> None:
    """重置全局 Database（用於測試）"""
    global _database
    
    with _database_lock:
        if _database is not None:
            _database.close()
        _database = None