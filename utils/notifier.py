"""
Notifier 模組 - 通知服務

提供多管道通知功能：Telegram、Email、Desktop
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, List, Any, Callable
import threading
from queue import Queue
import json

# 可選依賴
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import aiosmtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    HAS_AIOSMTPLIB = True
except ImportError:
    HAS_AIOSMTPLIB = False


# 設定 logger
logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """通知級別"""
    
    DEBUG = auto()      # 調試（不發送）
    INFO = auto()       # 資訊（可選發送）
    WARNING = auto()    # 警告
    ERROR = auto()      # 錯誤
    CRITICAL = auto()   # 緊急


class NotificationChannel(Enum):
    """通知管道"""
    
    TELEGRAM = auto()
    EMAIL = auto()
    DESKTOP = auto()
    WEBHOOK = auto()
    ALL = auto()


@dataclass
class NotificationConfig:
    """通知配置"""
    
    # Telegram 配置
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # Email 配置
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # Webhook 配置
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    
    # 通用設定
    min_level: NotificationLevel = NotificationLevel.WARNING
    rate_limit_seconds: int = 60  # 相同訊息的速率限制
    async_send: bool = True  # 是否異步發送
    
    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """從環境變數載入配置"""
        return cls(
            telegram_enabled=os.getenv("TELEGRAM_ENABLED", "").lower() == "true",
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            email_enabled=os.getenv("EMAIL_ENABLED", "").lower() == "true",
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.getenv("SMTP_USERNAME", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            smtp_use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
            email_from=os.getenv("EMAIL_FROM", ""),
            email_to=os.getenv("EMAIL_TO", "").split(",") if os.getenv("EMAIL_TO") else [],
            webhook_enabled=os.getenv("WEBHOOK_ENABLED", "").lower() == "true",
            webhook_url=os.getenv("WEBHOOK_URL", ""),
        )


@dataclass
class NotificationRecord:
    """通知記錄"""
    
    message: str
    level: NotificationLevel
    channel: NotificationChannel
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    error: str = ""


class Notifier:
    """
    通知服務
    
    提供統一的多管道通知功能
    
    使用方式:
        notifier = Notifier(config)
        await notifier.initialize()
        
        # 發送告警
        await notifier.alert("交易執行失敗", level=NotificationLevel.ERROR)
        
        # 發送到特定管道
        await notifier.send_telegram("測試訊息")
        await notifier.send_email("主題", "內容")
    """
    
    def __init__(
        self,
        config: Optional[NotificationConfig] = None,
    ):
        """
        初始化通知服務
        
        Args:
            config: 通知配置
        """
        self._config = config or NotificationConfig()
        
        # 狀態
        self._initialized = False
        self._telegram_session: Optional[aiohttp.ClientSession] = None
        
        # 速率限制
        self._last_messages: Dict[str, datetime] = {}
        
        # 通知歷史
        self._history: List[NotificationRecord] = []
        self._max_history = 100
        
        # 異步隊列
        self._queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 回調
        self._on_send_callbacks: List[Callable[[NotificationRecord], None]] = []
        
        logger.info("Notifier 初始化完成")
    
    # ========== 初始化 ==========
    
    async def initialize(self) -> None:
        """初始化通知服務"""
        if self._initialized:
            return
        
        # 初始化 Telegram
        if self._config.telegram_enabled and HAS_AIOHTTP:
            try:
                self._telegram_session = aiohttp.ClientSession()
                # 測試連接
                await self._test_telegram()
                logger.info("Telegram 通知已啟用")
            except Exception as e:
                logger.error(f"Telegram 初始化失敗: {e}")
                self._config.telegram_enabled = False
        
        # 啟動異步 worker
        if self._config.async_send:
            self._start_worker()
        
        self._initialized = True
        logger.info("Notifier 初始化完成")
    
    def initialize_sync(self) -> None:
        """同步初始化"""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.initialize())
        finally:
            loop.close()
    
    async def shutdown(self) -> None:
        """關閉通知服務"""
        self._running = False
        
        if self._telegram_session:
            await self._telegram_session.close()
            self._telegram_session = None
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._queue.put(None)  # 發送停止信號
            self._worker_thread.join(timeout=5)
        
        self._initialized = False
        logger.info("Notifier 已關閉")
    
    async def _test_telegram(self) -> bool:
        """測試 Telegram 連接"""
        if not self._telegram_session:
            return False
        
        url = f"https://api.telegram.org/bot{self._config.telegram_bot_token}/getMe"
        
        try:
            async with self._telegram_session.get(url) as response:
                data = await response.json()
                if data.get("ok"):
                    logger.debug(f"Telegram Bot: {data['result'].get('username')}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Telegram 測試失敗: {e}")
            return False
    
    # ========== Worker ==========
    
    def _start_worker(self) -> None:
        """啟動異步 worker"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def _worker_loop(self) -> None:
        """Worker 循環"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self._running:
            try:
                item = self._queue.get(timeout=1)
                if item is None:
                    break
                
                # 執行異步發送
                message, level, channel, kwargs = item
                loop.run_until_complete(
                    self._send_async(message, level, channel, **kwargs)
                )
            except Exception:
                pass  # Queue timeout
        
        loop.close()
    
    async def _send_async(
        self,
        message: str,
        level: NotificationLevel,
        channel: NotificationChannel,
        **kwargs,
    ) -> None:
        """異步發送（worker 調用）"""
        try:
            if channel == NotificationChannel.TELEGRAM:
                await self.send_telegram(message, **kwargs)
            elif channel == NotificationChannel.EMAIL:
                subject = kwargs.get("subject", f"[{level.name}] Trading Alert")
                await self.send_email(subject, message, **kwargs)
            elif channel == NotificationChannel.WEBHOOK:
                await self.send_webhook(message, **kwargs)
        except Exception as e:
            logger.error(f"異步通知發送失敗: {e}")
    
    # ========== 統一告警 ==========
    
    async def alert(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.WARNING,
        channel: Optional[NotificationChannel] = None,
        title: str = "",
        **kwargs,
    ) -> bool:
        """
        統一告警方法
        
        根據級別和配置選擇通知管道
        
        Args:
            message: 告警訊息
            level: 告警級別
            channel: 指定管道，None 則自動選擇
            title: 標題
            **kwargs: 額外參數
            
        Returns:
            是否成功發送
        """
        # 檢查級別
        if level.value < self._config.min_level.value:
            logger.debug(f"告警級別 {level.name} 低於最低級別，跳過")
            return False
        
        # 速率限制檢查
        if not self._check_rate_limit(message):
            logger.debug(f"訊息被速率限制: {message[:50]}...")
            return False
        
        # 格式化訊息
        formatted = self._format_message(message, level, title)
        
        # 選擇管道
        channels = self._select_channels(level, channel)
        
        success = False
        
        for ch in channels:
            try:
                if self._config.async_send and self._running:
                    # 放入隊列異步發送
                    self._queue.put((formatted, level, ch, kwargs))
                    success = True
                else:
                    # 同步發送
                    if ch == NotificationChannel.TELEGRAM:
                        success = await self.send_telegram(formatted, **kwargs) or success
                    elif ch == NotificationChannel.EMAIL:
                        subject = title or f"[{level.name}] Trading Alert"
                        success = await self.send_email(subject, formatted, **kwargs) or success
                    elif ch == NotificationChannel.WEBHOOK:
                        success = await self.send_webhook(formatted, **kwargs) or success
            except Exception as e:
                logger.error(f"發送到 {ch.name} 失敗: {e}")
                self._record_notification(formatted, level, ch, False, str(e))
        
        return success
    
    def alert_sync(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.WARNING,
        **kwargs,
    ) -> bool:
        """同步告警方法"""
        if self._config.async_send and self._running:
            # 使用隊列
            formatted = self._format_message(message, level, kwargs.get("title", ""))
            channels = self._select_channels(level, kwargs.get("channel"))
            
            for ch in channels:
                self._queue.put((formatted, level, ch, kwargs))
            return True
        else:
            # 建立新事件循環
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.alert(message, level, **kwargs))
            finally:
                loop.close()
    
    def _format_message(
        self,
        message: str,
        level: NotificationLevel,
        title: str = "",
    ) -> str:
        """格式化訊息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 級別圖標
        icons = {
            NotificationLevel.DEBUG: "🔧",
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
        }
        icon = icons.get(level, "📢")
        
        if title:
            return f"{icon} *{title}*\n\n{message}\n\n_{timestamp}_"
        else:
            return f"{icon} *[{level.name}]*\n\n{message}\n\n_{timestamp}_"
    
    def _select_channels(
        self,
        level: NotificationLevel,
        channel: Optional[NotificationChannel],
    ) -> List[NotificationChannel]:
        """選擇通知管道"""
        if channel == NotificationChannel.ALL:
            channels = []
            if self._config.telegram_enabled:
                channels.append(NotificationChannel.TELEGRAM)
            if self._config.email_enabled:
                channels.append(NotificationChannel.EMAIL)
            if self._config.webhook_enabled:
                channels.append(NotificationChannel.WEBHOOK)
            return channels
        
        if channel:
            return [channel]
        
        # 根據級別自動選擇
        channels = []
        
        if level in [NotificationLevel.CRITICAL, NotificationLevel.ERROR]:
            # 緊急：所有管道
            if self._config.telegram_enabled:
                channels.append(NotificationChannel.TELEGRAM)
            if self._config.email_enabled:
                channels.append(NotificationChannel.EMAIL)
        elif level == NotificationLevel.WARNING:
            # 警告：Telegram
            if self._config.telegram_enabled:
                channels.append(NotificationChannel.TELEGRAM)
        else:
            # 資訊：Telegram（如果啟用）
            if self._config.telegram_enabled:
                channels.append(NotificationChannel.TELEGRAM)
        
        return channels
    
    def _check_rate_limit(self, message: str) -> bool:
        """檢查速率限制"""
        # 使用訊息前 100 字元作為 key
        key = message[:100]
        now = datetime.now()
        
        if key in self._last_messages:
            elapsed = (now - self._last_messages[key]).total_seconds()
            if elapsed < self._config.rate_limit_seconds:
                return False
        
        self._last_messages[key] = now
        return True
    
    # ========== Telegram ==========
    
    async def send_telegram(
        self,
        message: str,
        chat_id: Optional[str] = None,
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
        **kwargs,
    ) -> bool:
        """
        發送 Telegram 訊息
        
        Args:
            message: 訊息內容
            chat_id: 聊天 ID，None 使用配置
            parse_mode: 解析模式 ("Markdown", "HTML")
            disable_notification: 是否靜音
            
        Returns:
            是否成功
        """
        if not self._config.telegram_enabled:
            logger.debug("Telegram 未啟用")
            return False
        
        if not HAS_AIOHTTP:
            logger.error("未安裝 aiohttp，無法發送 Telegram")
            return False
        
        chat_id = chat_id or self._config.telegram_chat_id
        if not chat_id:
            logger.error("未設定 Telegram chat_id")
            return False
        
        url = f"https://api.telegram.org/bot{self._config.telegram_bot_token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }
        
        try:
            # 建立新的 session 如果需要
            session = self._telegram_session
            close_session = False
            
            if session is None or session.closed:
                session = aiohttp.ClientSession()
                close_session = True
            
            try:
                async with session.post(url, json=payload) as response:
                    data = await response.json()
                    
                    if data.get("ok"):
                        logger.debug(f"Telegram 訊息已發送")
                        self._record_notification(
                            message, NotificationLevel.INFO, 
                            NotificationChannel.TELEGRAM, True
                        )
                        return True
                    else:
                        error = data.get("description", "Unknown error")
                        logger.error(f"Telegram 發送失敗: {error}")
                        self._record_notification(
                            message, NotificationLevel.INFO,
                            NotificationChannel.TELEGRAM, False, error
                        )
                        return False
            finally:
                if close_session:
                    await session.close()
                    
        except Exception as e:
            logger.error(f"Telegram 發送錯誤: {e}")
            self._record_notification(
                message, NotificationLevel.INFO,
                NotificationChannel.TELEGRAM, False, str(e)
            )
            return False
    
    # ========== Email ==========
    
    async def send_email(
        self,
        subject: str,
        body: str,
        to: Optional[List[str]] = None,
        html: bool = False,
        **kwargs,
    ) -> bool:
        """
        發送 Email
        
        Args:
            subject: 主題
            body: 內容
            to: 收件者列表，None 使用配置
            html: 是否 HTML 格式
            
        Returns:
            是否成功
        """
        if not self._config.email_enabled:
            logger.debug("Email 未啟用")
            return False
        
        if not HAS_AIOSMTPLIB:
            logger.error("未安裝 aiosmtplib，無法發送 Email")
            return False
        
        to = to or self._config.email_to
        if not to:
            logger.error("未設定 Email 收件者")
            return False
        
        try:
            # 建立郵件
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self._config.email_from or self._config.smtp_username
            msg["To"] = ", ".join(to)
            
            # 添加內容
            if html:
                msg.attach(MIMEText(body, "html"))
            else:
                msg.attach(MIMEText(body, "plain"))
            
            # 發送
            await aiosmtplib.send(
                msg,
                hostname=self._config.smtp_host,
                port=self._config.smtp_port,
                username=self._config.smtp_username,
                password=self._config.smtp_password,
                start_tls=self._config.smtp_use_tls,
            )
            
            logger.debug(f"Email 已發送: {subject}")
            self._record_notification(
                body, NotificationLevel.INFO,
                NotificationChannel.EMAIL, True
            )
            return True
            
        except Exception as e:
            logger.error(f"Email 發送錯誤: {e}")
            self._record_notification(
                body, NotificationLevel.INFO,
                NotificationChannel.EMAIL, False, str(e)
            )
            return False
    
    # ========== Webhook ==========
    
    async def send_webhook(
        self,
        message: str,
        url: Optional[str] = None,
        payload_format: str = "json",
        **kwargs,
    ) -> bool:
        """
        發送 Webhook
        
        Args:
            message: 訊息內容
            url: Webhook URL，None 使用配置
            payload_format: 格式 ("json", "form")
            
        Returns:
            是否成功
        """
        if not self._config.webhook_enabled:
            logger.debug("Webhook 未啟用")
            return False
        
        if not HAS_AIOHTTP:
            logger.error("未安裝 aiohttp，無法發送 Webhook")
            return False
        
        url = url or self._config.webhook_url
        if not url:
            logger.error("未設定 Webhook URL")
            return False
        
        try:
            payload = {
                "message": message,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }
            
            headers = {
                "Content-Type": "application/json",
                **self._config.webhook_headers,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status in [200, 201, 202, 204]:
                        logger.debug(f"Webhook 已發送")
                        self._record_notification(
                            message, NotificationLevel.INFO,
                            NotificationChannel.WEBHOOK, True
                        )
                        return True
                    else:
                        error = f"HTTP {response.status}"
                        logger.error(f"Webhook 發送失敗: {error}")
                        self._record_notification(
                            message, NotificationLevel.INFO,
                            NotificationChannel.WEBHOOK, False, error
                        )
                        return False
                        
        except Exception as e:
            logger.error(f"Webhook 發送錯誤: {e}")
            self._record_notification(
                message, NotificationLevel.INFO,
                NotificationChannel.WEBHOOK, False, str(e)
            )
            return False
    
    # ========== 記錄 ==========
    
    def _record_notification(
        self,
        message: str,
        level: NotificationLevel,
        channel: NotificationChannel,
        success: bool,
        error: str = "",
    ) -> None:
        """記錄通知"""
        record = NotificationRecord(
            message=message[:200],  # 截斷長訊息
            level=level,
            channel=channel,
            success=success,
            error=error,
        )
        
        self._history.append(record)
        
        # 限制歷史數量
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        # 執行回調
        for callback in self._on_send_callbacks:
            try:
                callback(record)
            except Exception as e:
                logger.error(f"通知回調錯誤: {e}")
    
    def get_history(self, limit: int = 50) -> List[NotificationRecord]:
        """取得通知歷史"""
        return self._history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """取得統計"""
        total = len(self._history)
        success = sum(1 for r in self._history if r.success)
        
        by_channel = {}
        for ch in NotificationChannel:
            ch_records = [r for r in self._history if r.channel == ch]
            if ch_records:
                by_channel[ch.name] = {
                    "total": len(ch_records),
                    "success": sum(1 for r in ch_records if r.success),
                }
        
        return {
            "total": total,
            "success": success,
            "failure": total - success,
            "success_rate": f"{success/total:.2%}" if total > 0 else "N/A",
            "by_channel": by_channel,
            "telegram_enabled": self._config.telegram_enabled,
            "email_enabled": self._config.email_enabled,
            "webhook_enabled": self._config.webhook_enabled,
        }
    
    # ========== 回調 ==========
    
    def on_send(self, callback: Callable[[NotificationRecord], None]) -> Callable:
        """註冊發送回調"""
        self._on_send_callbacks.append(callback)
        return callback


# ============================================================
# 便捷函數
# ============================================================

_notifier: Optional[Notifier] = None
_notifier_lock = threading.Lock()


def get_notifier(config: Optional[NotificationConfig] = None) -> Notifier:
    """取得全局 Notifier 實例"""
    global _notifier
    
    if _notifier is None:
        with _notifier_lock:
            if _notifier is None:
                _notifier = Notifier(config)
    
    return _notifier


def reset_notifier() -> None:
    """重置全局 Notifier"""
    global _notifier
    
    with _notifier_lock:
        if _notifier:
            asyncio.get_event_loop().run_until_complete(_notifier.shutdown())
        _notifier = None