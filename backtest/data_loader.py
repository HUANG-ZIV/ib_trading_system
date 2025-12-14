"""
DataLoader 模組 - 歷史數據載入器

提供多種數據來源的歷史數據載入功能
"""

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import asyncio

from core.events import BarEvent, EventType


# 設定 logger
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """數據來源"""
    
    CSV = auto()
    DATABASE = auto()
    IB = auto()
    YAHOO = auto()
    CUSTOM = auto()


class TimeFrame(Enum):
    """時間週期"""
    
    TICK = "tick"
    SEC_1 = "1 sec"
    SEC_5 = "5 secs"
    SEC_15 = "15 secs"
    SEC_30 = "30 secs"
    MIN_1 = "1 min"
    MIN_2 = "2 mins"
    MIN_3 = "3 mins"
    MIN_5 = "5 mins"
    MIN_15 = "15 mins"
    MIN_30 = "30 mins"
    HOUR_1 = "1 hour"
    HOUR_2 = "2 hours"
    HOUR_4 = "4 hours"
    DAY_1 = "1 day"
    WEEK_1 = "1 week"
    MONTH_1 = "1 month"


@dataclass
class HistoricalBar:
    """歷史 K 線數據"""
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    
    # 額外欄位
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    
    # 調整後價格
    adj_close: Optional[float] = None
    
    def to_bar_event(self, symbol: str, timeframe: str = "1d") -> BarEvent:
        """轉換為 BarEvent"""
        return BarEvent(
            event_type=EventType.BAR,
            symbol=symbol,
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            vwap=self.vwap,
            trade_count=self.trade_count,
            timeframe=timeframe,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "adj_close": self.adj_close,
        }
    
    def is_valid(self) -> bool:
        """驗證數據有效性"""
        # 價格必須為正
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            return False
        
        # high 必須是最高
        if self.high < self.open or self.high < self.close or self.high < self.low:
            return False
        
        # low 必須是最低
        if self.low > self.open or self.low > self.close or self.low > self.high:
            return False
        
        return True


@dataclass
class DataLoadResult:
    """數據載入結果"""
    
    symbol: str
    bars: List[HistoricalBar] = field(default_factory=list)
    source: DataSource = DataSource.CSV
    timeframe: str = "1d"
    
    # 統計
    total_bars: int = 0
    valid_bars: int = 0
    invalid_bars: int = 0
    filled_bars: int = 0  # 填補的缺失值
    
    # 時間範圍
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # 錯誤
    errors: List[str] = field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        """是否成功"""
        return len(self.bars) > 0 and len(self.errors) == 0
    
    def to_bar_events(self) -> List[BarEvent]:
        """轉換為 BarEvent 列表"""
        return [bar.to_bar_event(self.symbol, self.timeframe) for bar in self.bars]


class DataLoader:
    """
    歷史數據載入器
    
    支援從多種來源載入歷史數據
    
    使用方式:
        loader = DataLoader()
        
        # 從 CSV 載入
        result = loader.load_from_csv("AAPL", "data/aapl.csv")
        
        # 從 IB 載入
        result = await loader.load_from_ib(
            "AAPL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            timeframe=TimeFrame.DAY_1,
        )
        
        # 轉換為 BarEvent
        bars = result.to_bar_events()
    """
    
    def __init__(
        self,
        validate_data: bool = True,
        fill_missing: bool = True,
        remove_invalid: bool = True,
    ):
        """
        初始化數據載入器
        
        Args:
            validate_data: 是否驗證數據
            fill_missing: 是否填補缺失值
            remove_invalid: 是否移除無效數據
        """
        self._validate_data = validate_data
        self._fill_missing = fill_missing
        self._remove_invalid = remove_invalid
        
        # IB 連接（延遲初始化）
        self._ib = None
        
        logger.info("DataLoader 初始化完成")
    
    # ========== CSV 載入 ==========
    
    def load_from_csv(
        self,
        symbol: str,
        filepath: Union[str, Path],
        timeframe: str = "1d",
        date_column: str = "date",
        date_format: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        encoding: str = "utf-8",
    ) -> DataLoadResult:
        """
        從 CSV 檔案載入數據
        
        Args:
            symbol: 標的代碼
            filepath: CSV 檔案路徑
            timeframe: 時間週期
            date_column: 日期欄位名稱
            date_format: 日期格式，None 則自動偵測
            column_mapping: 欄位映射 {"csv欄位": "標準欄位"}
            start_date: 開始日期
            end_date: 結束日期
            encoding: 檔案編碼
            
        Returns:
            DataLoadResult
        """
        result = DataLoadResult(
            symbol=symbol,
            source=DataSource.CSV,
            timeframe=timeframe,
        )
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            result.errors.append(f"檔案不存在: {filepath}")
            return result
        
        # 預設欄位映射
        default_mapping = {
            "date": "timestamp",
            "datetime": "timestamp",
            "time": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "vol": "volume",
            "adj close": "adj_close",
            "adjusted close": "adj_close",
            "vwap": "vwap",
        }
        
        if column_mapping:
            default_mapping.update(column_mapping)
        
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                
                # 標準化欄位名稱
                fieldnames = {name.lower().strip(): name for name in reader.fieldnames}
                
                for row in reader:
                    try:
                        bar = self._parse_csv_row(
                            row, fieldnames, default_mapping, date_column, date_format
                        )
                        
                        if bar is None:
                            result.invalid_bars += 1
                            continue
                        
                        # 日期過濾
                        if start_date and bar.timestamp.date() < start_date:
                            continue
                        if end_date and bar.timestamp.date() > end_date:
                            continue
                        
                        result.bars.append(bar)
                        result.total_bars += 1
                        
                    except Exception as e:
                        result.invalid_bars += 1
                        logger.debug(f"解析行錯誤: {e}")
            
            # 排序
            result.bars.sort(key=lambda x: x.timestamp)
            
            # 驗證和處理
            if self._validate_data:
                result = self._validate_and_clean(result)
            
            # 填補缺失
            if self._fill_missing:
                result = self._fill_missing_bars(result)
            
            # 更新統計
            if result.bars:
                result.start_date = result.bars[0].timestamp
                result.end_date = result.bars[-1].timestamp
                result.valid_bars = len(result.bars)
            
            logger.info(f"從 CSV 載入 {symbol}: {len(result.bars)} 根 K 線")
            
        except Exception as e:
            result.errors.append(f"讀取檔案錯誤: {e}")
            logger.error(f"CSV 載入錯誤: {e}")
        
        return result
    
    def _parse_csv_row(
        self,
        row: Dict[str, str],
        fieldnames: Dict[str, str],
        mapping: Dict[str, str],
        date_column: str,
        date_format: Optional[str],
    ) -> Optional[HistoricalBar]:
        """解析 CSV 行"""
        # 標準化行的鍵
        row_lower = {k.lower().strip(): v for k, v in row.items()}
        
        # 找到日期欄位
        timestamp = None
        for col in [date_column.lower(), "date", "datetime", "time", "timestamp"]:
            if col in row_lower:
                timestamp = self._parse_datetime(row_lower[col], date_format)
                break
        
        if timestamp is None:
            return None
        
        # 解析價格
        def get_float(keys: List[str]) -> Optional[float]:
            for key in keys:
                if key in row_lower and row_lower[key]:
                    try:
                        return float(row_lower[key].replace(',', ''))
                    except ValueError:
                        pass
            return None
        
        def get_int(keys: List[str]) -> int:
            for key in keys:
                if key in row_lower and row_lower[key]:
                    try:
                        return int(float(row_lower[key].replace(',', '')))
                    except ValueError:
                        pass
            return 0
        
        open_price = get_float(["open", "o"])
        high_price = get_float(["high", "h"])
        low_price = get_float(["low", "l"])
        close_price = get_float(["close", "c", "price"])
        volume = get_int(["volume", "vol", "v"])
        
        if None in [open_price, high_price, low_price, close_price]:
            return None
        
        return HistoricalBar(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            adj_close=get_float(["adj close", "adjusted close", "adj_close"]),
            vwap=get_float(["vwap"]),
        )
    
    def _parse_datetime(
        self,
        value: str,
        date_format: Optional[str] = None,
    ) -> Optional[datetime]:
        """解析日期時間"""
        if not value:
            return None
        
        value = value.strip()
        
        # 指定格式
        if date_format:
            try:
                return datetime.strptime(value, date_format)
            except ValueError:
                pass
        
        # 自動偵測常見格式
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
            "%Y%m%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        return None
    
    # ========== 資料庫載入 ==========
    
    def load_from_database(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        db_url: Optional[str] = None,
        table_name: str = "bars",
    ) -> DataLoadResult:
        """
        從資料庫載入數據
        
        Args:
            symbol: 標的代碼
            timeframe: 時間週期
            start_date: 開始日期
            end_date: 結束日期
            db_url: 資料庫連接字串
            table_name: 資料表名稱
            
        Returns:
            DataLoadResult
        """
        result = DataLoadResult(
            symbol=symbol,
            source=DataSource.DATABASE,
            timeframe=timeframe,
        )
        
        try:
            from sqlalchemy import create_engine, text
            
            # 使用預設或指定的資料庫 URL
            if db_url is None:
                db_url = "sqlite:///data/trading.db"
            
            engine = create_engine(db_url)
            
            # 建立查詢
            query = f"""
                SELECT timestamp, open, high, low, close, volume, vwap
                FROM {table_name}
                WHERE symbol = :symbol
                AND timeframe = :timeframe
            """
            
            params = {"symbol": symbol, "timeframe": timeframe}
            
            if start_date:
                query += " AND timestamp >= :start_date"
                params["start_date"] = datetime.combine(start_date, datetime.min.time())
            
            if end_date:
                query += " AND timestamp <= :end_date"
                params["end_date"] = datetime.combine(end_date, datetime.max.time())
            
            query += " ORDER BY timestamp"
            
            with engine.connect() as conn:
                rows = conn.execute(text(query), params)
                
                for row in rows:
                    bar = HistoricalBar(
                        timestamp=row.timestamp,
                        open=row.open,
                        high=row.high,
                        low=row.low,
                        close=row.close,
                        volume=row.volume or 0,
                        vwap=row.vwap,
                    )
                    result.bars.append(bar)
                    result.total_bars += 1
            
            # 驗證和處理
            if self._validate_data:
                result = self._validate_and_clean(result)
            
            # 更新統計
            if result.bars:
                result.start_date = result.bars[0].timestamp
                result.end_date = result.bars[-1].timestamp
                result.valid_bars = len(result.bars)
            
            logger.info(f"從資料庫載入 {symbol}: {len(result.bars)} 根 K 線")
            
        except ImportError:
            result.errors.append("未安裝 SQLAlchemy")
        except Exception as e:
            result.errors.append(f"資料庫錯誤: {e}")
            logger.error(f"資料庫載入錯誤: {e}")
        
        return result
    
    # ========== IB 載入 ==========
    
    async def load_from_ib(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        ib_connection=None,
    ) -> DataLoadResult:
        """
        從 IB 下載歷史數據
        
        Args:
            symbol: 標的代碼
            start_date: 開始日期
            end_date: 結束日期
            timeframe: 時間週期
            what_to_show: 數據類型 (TRADES, MIDPOINT, BID, ASK)
            use_rth: 是否只用正常交易時段
            ib_connection: IB 連接，None 則使用內部連接
            
        Returns:
            DataLoadResult
        """
        result = DataLoadResult(
            symbol=symbol,
            source=DataSource.IB,
            timeframe=timeframe.value,
        )
        
        try:
            from ib_insync import IB, Stock, util
            
            # 使用提供的連接或建立新連接
            ib = ib_connection
            should_disconnect = False
            
            if ib is None:
                ib = IB()
                await ib.connectAsync('127.0.0.1', 7497, clientId=999)
                should_disconnect = True
            
            try:
                # 建立合約
                contract = Stock(symbol, 'SMART', 'USD')
                await ib.qualifyContractsAsync(contract)
                
                # 計算持續時間
                if end_date is None:
                    end_date = date.today()
                
                if start_date is None:
                    start_date = end_date - timedelta(days=365)
                
                duration_days = (end_date - start_date).days
                
                # IB 歷史數據請求有限制，可能需要分批
                duration_str = f"{duration_days} D"
                
                # 下載數據
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=datetime.combine(end_date, datetime.max.time()),
                    durationStr=duration_str,
                    barSizeSetting=timeframe.value,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=1,
                )
                
                # 轉換數據
                for bar in bars:
                    historical_bar = HistoricalBar(
                        timestamp=bar.date if isinstance(bar.date, datetime) else datetime.combine(bar.date, datetime.min.time()),
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=int(bar.volume),
                        vwap=getattr(bar, 'average', None),
                        trade_count=getattr(bar, 'barCount', None),
                    )
                    result.bars.append(historical_bar)
                    result.total_bars += 1
                
            finally:
                if should_disconnect:
                    ib.disconnect()
            
            # 驗證和處理
            if self._validate_data:
                result = self._validate_and_clean(result)
            
            # 更新統計
            if result.bars:
                result.start_date = result.bars[0].timestamp
                result.end_date = result.bars[-1].timestamp
                result.valid_bars = len(result.bars)
            
            logger.info(f"從 IB 載入 {symbol}: {len(result.bars)} 根 K 線")
            
        except ImportError:
            result.errors.append("未安裝 ib_insync")
        except Exception as e:
            result.errors.append(f"IB 錯誤: {e}")
            logger.error(f"IB 載入錯誤: {e}")
        
        return result
    
    def load_from_ib_sync(
        self,
        symbol: str,
        **kwargs,
    ) -> DataLoadResult:
        """同步版本的 IB 載入"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.load_from_ib(symbol, **kwargs))
        finally:
            loop.close()
    
    # ========== Yahoo Finance 載入 ==========
    
    def load_from_yahoo(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        interval: str = "1d",
    ) -> DataLoadResult:
        """
        從 Yahoo Finance 載入數據
        
        Args:
            symbol: 標的代碼
            start_date: 開始日期
            end_date: 結束日期
            interval: 時間間隔 (1d, 1wk, 1mo, 1h, etc.)
            
        Returns:
            DataLoadResult
        """
        result = DataLoadResult(
            symbol=symbol,
            source=DataSource.YAHOO,
            timeframe=interval,
        )
        
        try:
            import yfinance as yf
            
            # 預設日期範圍
            if end_date is None:
                end_date = date.today()
            if start_date is None:
                start_date = end_date - timedelta(days=365)
            
            # 下載數據
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
            )
            
            if df.empty:
                result.errors.append("沒有取得數據")
                return result
            
            # 轉換數據
            for idx, row in df.iterrows():
                timestamp = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx
                
                bar = HistoricalBar(
                    timestamp=timestamp,
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=int(row['Volume']),
                )
                result.bars.append(bar)
                result.total_bars += 1
            
            # 驗證和處理
            if self._validate_data:
                result = self._validate_and_clean(result)
            
            # 更新統計
            if result.bars:
                result.start_date = result.bars[0].timestamp
                result.end_date = result.bars[-1].timestamp
                result.valid_bars = len(result.bars)
            
            logger.info(f"從 Yahoo 載入 {symbol}: {len(result.bars)} 根 K 線")
            
        except ImportError:
            result.errors.append("未安裝 yfinance")
        except Exception as e:
            result.errors.append(f"Yahoo 錯誤: {e}")
            logger.error(f"Yahoo 載入錯誤: {e}")
        
        return result
    
    # ========== 數據處理 ==========
    
    def _validate_and_clean(self, result: DataLoadResult) -> DataLoadResult:
        """驗證並清理數據"""
        if not result.bars:
            return result
        
        valid_bars = []
        
        for bar in result.bars:
            if bar.is_valid():
                valid_bars.append(bar)
            else:
                result.invalid_bars += 1
                
                if not self._remove_invalid:
                    # 嘗試修復
                    fixed_bar = self._fix_invalid_bar(bar)
                    if fixed_bar:
                        valid_bars.append(fixed_bar)
        
        result.bars = valid_bars
        return result
    
    def _fix_invalid_bar(self, bar: HistoricalBar) -> Optional[HistoricalBar]:
        """嘗試修復無效數據"""
        # 確保 high >= all others
        bar.high = max(bar.open, bar.high, bar.low, bar.close)
        
        # 確保 low <= all others
        bar.low = min(bar.open, bar.high, bar.low, bar.close)
        
        # 驗證修復結果
        if bar.is_valid():
            return bar
        
        return None
    
    def _fill_missing_bars(self, result: DataLoadResult) -> DataLoadResult:
        """填補缺失的 K 線"""
        if len(result.bars) < 2:
            return result
        
        filled_bars = [result.bars[0]]
        
        for i in range(1, len(result.bars)):
            current = result.bars[i]
            previous = filled_bars[-1]
            
            # 檢查時間差
            time_diff = current.timestamp - previous.timestamp
            
            # 對於日線，檢查是否缺少交易日
            # 這裡簡化處理，不考慮假日
            expected_diff = timedelta(days=1)
            
            if result.timeframe == "1d" and time_diff > expected_diff * 3:
                # 可能缺少數據，使用前一根的收盤價填補
                # 實際應用中可能需要更複雜的處理
                logger.debug(f"偵測到數據缺口: {previous.timestamp} -> {current.timestamp}")
            
            filled_bars.append(current)
        
        result.bars = filled_bars
        return result
    
    # ========== 批次載入 ==========
    
    def load_multiple_from_csv(
        self,
        symbols: List[str],
        data_dir: Union[str, Path],
        filename_pattern: str = "{symbol}.csv",
        **kwargs,
    ) -> Dict[str, DataLoadResult]:
        """
        批次從 CSV 載入多個標的
        
        Args:
            symbols: 標的列表
            data_dir: 數據目錄
            filename_pattern: 檔案名稱模式
            **kwargs: 傳遞給 load_from_csv 的參數
            
        Returns:
            {symbol: DataLoadResult}
        """
        results = {}
        data_dir = Path(data_dir)
        
        for symbol in symbols:
            filename = filename_pattern.format(symbol=symbol)
            filepath = data_dir / filename
            
            results[symbol] = self.load_from_csv(symbol, filepath, **kwargs)
        
        return results
    
    async def load_multiple_from_ib(
        self,
        symbols: List[str],
        **kwargs,
    ) -> Dict[str, DataLoadResult]:
        """
        批次從 IB 載入多個標的
        
        Args:
            symbols: 標的列表
            **kwargs: 傳遞給 load_from_ib 的參數
            
        Returns:
            {symbol: DataLoadResult}
        """
        results = {}
        
        try:
            from ib_insync import IB
            
            # 建立共享連接
            ib = IB()
            await ib.connectAsync('127.0.0.1', 7497, clientId=999)
            
            try:
                for symbol in symbols:
                    result = await self.load_from_ib(
                        symbol, ib_connection=ib, **kwargs
                    )
                    results[symbol] = result
                    
                    # 避免請求過快
                    await asyncio.sleep(1)
            finally:
                ib.disconnect()
                
        except Exception as e:
            logger.error(f"批次 IB 載入錯誤: {e}")
        
        return results
    
    # ========== 工具方法 ==========
    
    def merge_results(
        self,
        results: Dict[str, DataLoadResult],
    ) -> Dict[str, List[BarEvent]]:
        """
        合併多個載入結果為 BarEvent 字典
        
        Args:
            results: {symbol: DataLoadResult}
            
        Returns:
            {symbol: [BarEvent, ...]}
        """
        merged = {}
        
        for symbol, result in results.items():
            if result.is_success:
                merged[symbol] = result.to_bar_events()
        
        return merged
    
    def get_summary(self, results: Dict[str, DataLoadResult]) -> Dict[str, Any]:
        """取得載入摘要"""
        total_bars = 0
        total_errors = 0
        symbols_loaded = []
        symbols_failed = []
        
        for symbol, result in results.items():
            if result.is_success:
                symbols_loaded.append(symbol)
                total_bars += len(result.bars)
            else:
                symbols_failed.append(symbol)
                total_errors += len(result.errors)
        
        return {
            "symbols_loaded": len(symbols_loaded),
            "symbols_failed": len(symbols_failed),
            "total_bars": total_bars,
            "total_errors": total_errors,
            "loaded_symbols": symbols_loaded,
            "failed_symbols": symbols_failed,
        }