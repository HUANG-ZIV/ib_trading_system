"""
Settings 模組 - 全局配置管理

使用 dataclass 定義配置結構，從環境變數載入設定
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()


@dataclass
class IBConfig:
    """IB 連接配置"""
    
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    readonly: bool = False
    timeout: int = 30
    
    # 重連設定
    max_reconnect_attempts: int = 5
    reconnect_interval: int = 10  # 秒


@dataclass
class DatabaseConfig:
    """數據庫配置"""
    
    db_type: str = "sqlite"  # sqlite / duckdb
    db_path: str = "./data_store/trading.db"
    
    # 連接池設定
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class LogConfig:
    """日誌配置"""
    
    level: str = "INFO"
    path: str = "./logs"
    rotation: str = "10 MB"
    retention: str = "30 days"
    
    # 異步日誌（高頻交易用）
    async_logging: bool = True


@dataclass
class RiskConfig:
    """風控配置"""
    
    # 虧損限制
    max_daily_loss: float = 1000.0
    
    # 持倉限制
    max_position_size: int = 100
    max_position_per_symbol: int = 50
    max_total_exposure: float = 10000.0
    
    # 熔斷機制
    circuit_breaker_threshold: int = 5  # 連續虧損次數
    circuit_breaker_cooldown: int = 300  # 冷卻時間（秒）


@dataclass
class NotificationConfig:
    """通知配置"""
    
    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Email
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email: Optional[str] = None


@dataclass
class Settings:
    """主配置類，包含所有子配置"""
    
    # 交易模式: paper / live
    trading_mode: str = "paper"
    
    # 時區
    timezone: str = "America/New_York"
    
    # 子配置
    ib: IBConfig = field(default_factory=IBConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    log: LogConfig = field(default_factory=LogConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """從環境變數載入配置"""
        
        # IB 配置
        ib_config = IBConfig(
            host=os.getenv("IB_HOST", "127.0.0.1"),
            port=int(os.getenv("IB_PORT", "7497")),
            client_id=int(os.getenv("IB_CLIENT_ID", "1")),
            readonly=os.getenv("IB_READONLY", "False").lower() == "true",
            timeout=int(os.getenv("IB_TIMEOUT", "30")),
            max_reconnect_attempts=int(os.getenv("IB_MAX_RECONNECT_ATTEMPTS", "5")),
            reconnect_interval=int(os.getenv("IB_RECONNECT_INTERVAL", "10")),
        )
        
        # 數據庫配置
        database_config = DatabaseConfig(
            db_type=os.getenv("DB_TYPE", "sqlite"),
            db_path=os.getenv("DB_PATH", "./data_store/trading.db"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        )
        
        # 日誌配置
        log_config = LogConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            path=os.getenv("LOG_PATH", "./logs"),
            rotation=os.getenv("LOG_ROTATION", "10 MB"),
            retention=os.getenv("LOG_RETENTION", "30 days"),
            async_logging=os.getenv("LOG_ASYNC", "True").lower() == "true",
        )
        
        # 風控配置
        risk_config = RiskConfig(
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "1000")),
            max_position_size=int(os.getenv("MAX_POSITION_SIZE", "100")),
            max_position_per_symbol=int(os.getenv("MAX_POSITION_PER_SYMBOL", "50")),
            max_total_exposure=float(os.getenv("MAX_TOTAL_EXPOSURE", "10000")),
            circuit_breaker_threshold=int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
            circuit_breaker_cooldown=int(os.getenv("CIRCUIT_BREAKER_COOLDOWN", "300")),
        )
        
        # 通知配置
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        smtp_user = os.getenv("SMTP_USER")
        
        notification_config = NotificationConfig(
            telegram_enabled=bool(telegram_token),
            telegram_bot_token=telegram_token,
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            email_enabled=bool(smtp_user),
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=smtp_user,
            smtp_password=os.getenv("SMTP_PASSWORD"),
            alert_email=os.getenv("ALERT_EMAIL"),
        )
        
        return cls(
            trading_mode=os.getenv("TRADING_MODE", "paper"),
            timezone=os.getenv("TIMEZONE", "America/New_York"),
            ib=ib_config,
            database=database_config,
            log=log_config,
            risk=risk_config,
            notification=notification_config,
        )
    
    def ensure_directories(self) -> None:
        """確保必要的目錄存在"""
        # 日誌目錄
        Path(self.log.path).mkdir(parents=True, exist_ok=True)
        
        # 數據庫目錄
        Path(self.database.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def is_paper_trading(self) -> bool:
        """是否為模擬交易模式"""
        return self.trading_mode.lower() == "paper"
    
    def is_live_trading(self) -> bool:
        """是否為實盤交易模式"""
        return self.trading_mode.lower() == "live"
    
    def validate(self) -> None:
        """驗證配置有效性"""
        # 驗證 IB 端口
        valid_ports = [7496, 7497, 4001, 4002]
        if self.ib.port not in valid_ports:
            raise ValueError(f"IB port must be one of {valid_ports}, got {self.ib.port}")
        
        # 驗證風控參數
        if self.risk.max_daily_loss <= 0:
            raise ValueError("max_daily_loss must be positive")
        
        if self.risk.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")
        
        # 實盤模式警告
        if self.is_live_trading():
            import warnings
            warnings.warn("Running in LIVE trading mode!", UserWarning)


# 全局設定實例（單例）
_settings: Optional[Settings] = None


@lru_cache()
def get_settings() -> Settings:
    """
    取得全局設定實例（單例模式）
    
    使用 lru_cache 確保只創建一次
    """
    settings = Settings.from_env()
    settings.ensure_directories()
    return settings


def reload_settings() -> Settings:
    """
    重新載入設定
    
    清除快取並重新從環境變數載入
    """
    get_settings.cache_clear()
    return get_settings()


# 全局設定實例（方便導入使用）
settings = get_settings()