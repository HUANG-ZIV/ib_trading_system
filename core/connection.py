"""
Connection 模組 - IB 連接管理

使用 ib_insync 連接 TWS/Gateway，提供自動重連、狀態管理等功能
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Callable, Any, List
import threading
import time

from ib_insync import IB, Contract, Order, Trade, util

from .events import SystemEvent, SystemEventType, EventType
from .event_bus import EventBus, get_event_bus


# 設定 logger
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """連接狀態枚舉"""
    
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()


@dataclass
class ConnectionConfig:
    """連接配置"""
    
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    readonly: bool = False
    timeout: int = 30
    
    # 重連設定
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_interval: int = 10  # 秒


class IBConnection:
    """
    IB 連接管理器
    
    封裝 ib_insync 的 IB 類，提供：
    - 同步/異步連接方法
    - 自動重連機制
    - 連接狀態事件發布
    - 錯誤處理
    
    使用方式:
        # 方式一：直接使用
        conn = IBConnection(config)
        await conn.connect_async()
        # ... 使用連接
        await conn.disconnect_async()
        
        # 方式二：Context Manager
        async with ib_connection(config) as conn:
            # ... 使用連接
    """
    
    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        初始化 IB 連接管理器
        
        Args:
            config: 連接配置，None 使用預設值
            event_bus: 事件總線，None 使用全局單例
        """
        self._config = config or ConnectionConfig()
        self._event_bus = event_bus or get_event_bus()
        
        # ib_insync IB 實例
        self._ib = IB()
        
        # 連接狀態
        self._state = ConnectionState.DISCONNECTED
        self._connected_time: Optional[datetime] = None
        self._reconnect_count = 0
        
        # 重連控制
        self._reconnect_task: Optional[asyncio.Task] = None
        self._stop_reconnect = False
        
        # 回調函數
        self._on_connected_callbacks: List[Callable] = []
        self._on_disconnected_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []
        
        # 註冊 ib_insync 事件
        self._setup_ib_events()
        
        logger.debug(
            f"IBConnection 初始化: {self._config.host}:{self._config.port} "
            f"(client_id={self._config.client_id})"
        )
    
    # ========== 屬性 ==========
    
    @property
    def ib(self) -> IB:
        """取得底層 ib_insync IB 實例"""
        return self._ib
    
    @property
    def state(self) -> ConnectionState:
        """取得當前連接狀態"""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """是否已連接"""
        return self._ib.isConnected()
    
    @property
    def connected_time(self) -> Optional[datetime]:
        """連接建立時間"""
        return self._connected_time
    
    @property
    def reconnect_count(self) -> int:
        """重連次數"""
        return self._reconnect_count
    
    @property
    def config(self) -> ConnectionConfig:
        """取得連接配置"""
        return self._config
    
    # ========== 連接方法 ==========
    
    def connect(self) -> bool:
        """
        同步連接到 IB
        
        Returns:
            是否連接成功
        """
        if self.is_connected:
            logger.warning("已經連接到 IB")
            return True
        
        self._state = ConnectionState.CONNECTING
        self._emit_system_event(SystemEventType.INFO, "正在連接到 IB...")
        
        try:
            self._ib.connect(
                host=self._config.host,
                port=self._config.port,
                clientId=self._config.client_id,
                readonly=self._config.readonly,
                timeout=self._config.timeout,
            )
            
            self._on_connected()
            return True
            
        except Exception as e:
            self._on_connection_error(e)
            return False
    
    async def connect_async(self) -> bool:
        """
        異步連接到 IB
        
        Returns:
            是否連接成功
        """
        if self.is_connected:
            logger.warning("已經連接到 IB")
            return True
        
        self._state = ConnectionState.CONNECTING
        self._emit_system_event(SystemEventType.INFO, "正在連接到 IB...")
        
        try:
            await self._ib.connectAsync(
                host=self._config.host,
                port=self._config.port,
                clientId=self._config.client_id,
                readonly=self._config.readonly,
                timeout=self._config.timeout,
            )
            
            self._on_connected()
            return True
            
        except Exception as e:
            self._on_connection_error(e)
            return False
    
    def disconnect(self) -> None:
        """同步斷開連接"""
        self._stop_reconnect = True
        
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        
        if self.is_connected:
            self._ib.disconnect()
            self._on_disconnected()
    
    async def disconnect_async(self) -> None:
        """異步斷開連接"""
        self._stop_reconnect = True
        
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None
        
        if self.is_connected:
            self._ib.disconnect()
            self._on_disconnected()
    
    # ========== 重連機制 ==========
    
    async def _auto_reconnect(self) -> None:
        """自動重連邏輯"""
        if not self._config.auto_reconnect:
            return
        
        self._stop_reconnect = False
        self._state = ConnectionState.RECONNECTING
        
        while (
            not self._stop_reconnect
            and self._reconnect_count < self._config.max_reconnect_attempts
        ):
            self._reconnect_count += 1
            
            self._emit_system_event(
                SystemEventType.RECONNECTING,
                f"嘗試重連 ({self._reconnect_count}/{self._config.max_reconnect_attempts})..."
            )
            
            logger.info(
                f"嘗試重連到 IB "
                f"({self._reconnect_count}/{self._config.max_reconnect_attempts})"
            )
            
            # 等待一段時間再重連
            await asyncio.sleep(self._config.reconnect_interval)
            
            if self._stop_reconnect:
                break
            
            try:
                await self._ib.connectAsync(
                    host=self._config.host,
                    port=self._config.port,
                    clientId=self._config.client_id,
                    readonly=self._config.readonly,
                    timeout=self._config.timeout,
                )
                
                self._on_connected()
                self._reconnect_count = 0
                return
                
            except Exception as e:
                logger.warning(f"重連失敗: {e}")
        
        if not self._stop_reconnect:
            self._state = ConnectionState.ERROR
            self._emit_system_event(
                SystemEventType.ERROR,
                f"重連失敗，已達最大嘗試次數 ({self._config.max_reconnect_attempts})"
            )
            logger.error(f"重連失敗，已達最大嘗試次數")
    
    def reset_reconnect_count(self) -> None:
        """重置重連計數"""
        self._reconnect_count = 0
    
    # ========== 事件處理 ==========
    
    def _setup_ib_events(self) -> None:
        """設置 ib_insync 事件監聽"""
        self._ib.connectedEvent += self._handle_connected
        self._ib.disconnectedEvent += self._handle_disconnected
        self._ib.errorEvent += self._handle_error
    
    def _handle_connected(self) -> None:
        """處理連接成功事件"""
        # 由 connect/connect_async 方法處理
        pass
    
    def _handle_disconnected(self) -> None:
        """處理斷開連接事件"""
        if self._state == ConnectionState.CONNECTED:
            logger.warning("與 IB 的連接已斷開")
            self._state = ConnectionState.DISCONNECTED
            self._emit_system_event(SystemEventType.DISCONNECTED, "與 IB 的連接已斷開")
            
            # 執行回調
            for callback in self._on_disconnected_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"斷開連接回調錯誤: {e}")
            
            # 觸發自動重連
            if self._config.auto_reconnect and not self._stop_reconnect:
                try:
                    loop = asyncio.get_running_loop()
                    self._reconnect_task = loop.create_task(self._auto_reconnect())
                except RuntimeError:
                    # 沒有運行中的事件循環
                    logger.warning("無法啟動自動重連（沒有事件循環）")
    
    def _handle_error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        contract: Optional[Contract] = None,
    ) -> None:
        """處理錯誤事件"""
        # 過濾非關鍵錯誤
        # 2104, 2106, 2158 是連接成功的資訊
        # 2103 是市場數據連接斷開
        info_codes = {2104, 2106, 2158}
        warning_codes = {2103, 2105, 2107}
        
        if errorCode in info_codes:
            logger.debug(f"IB 資訊 [{errorCode}]: {errorString}")
            return
        
        if errorCode in warning_codes:
            logger.warning(f"IB 警告 [{errorCode}]: {errorString}")
            self._emit_system_event(
                SystemEventType.WARNING,
                errorString,
                error_code=errorCode,
            )
            return
        
        # 其他錯誤
        logger.error(f"IB 錯誤 [{errorCode}]: {errorString}")
        self._emit_system_event(
            SystemEventType.ERROR,
            errorString,
            error_code=errorCode,
        )
        
        # 執行錯誤回調
        for callback in self._on_error_callbacks:
            try:
                callback(errorCode, errorString)
            except Exception as e:
                logger.error(f"錯誤回調執行失敗: {e}")
    
    def _on_connected(self) -> None:
        """連接成功處理"""
        self._state = ConnectionState.CONNECTED
        self._connected_time = datetime.now()
        self._reconnect_count = 0
        
        logger.info(
            f"成功連接到 IB: {self._config.host}:{self._config.port} "
            f"(client_id={self._config.client_id})"
        )
        
        self._emit_system_event(
            SystemEventType.CONNECTED,
            f"成功連接到 IB (client_id={self._config.client_id})"
        )
        
        # 執行回調
        for callback in self._on_connected_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"連接回調錯誤: {e}")
    
    def _on_disconnected(self) -> None:
        """斷開連接處理"""
        self._state = ConnectionState.DISCONNECTED
        self._connected_time = None
        
        logger.info("已斷開與 IB 的連接")
        self._emit_system_event(SystemEventType.DISCONNECTED, "已斷開與 IB 的連接")
    
    def _on_connection_error(self, error: Exception) -> None:
        """連接錯誤處理"""
        self._state = ConnectionState.ERROR
        
        logger.error(f"連接到 IB 失敗: {error}")
        self._emit_system_event(
            SystemEventType.ERROR,
            f"連接到 IB 失敗: {error}"
        )
    
    def _emit_system_event(
        self,
        sub_type: SystemEventType,
        message: str,
        error_code: Optional[int] = None,
    ) -> None:
        """發布系統事件"""
        event = SystemEvent(
            event_type=EventType.SYSTEM,
            sub_type=sub_type,
            message=message,
            error_code=error_code,
            source="connection",
        )
        self._event_bus.publish(event)
    
    # ========== 回調註冊 ==========
    
    def on_connected(self, callback: Callable) -> Callable:
        """
        註冊連接成功回調
        
        可作為裝飾器使用:
            @conn.on_connected
            def handle_connected():
                print("已連接")
        """
        self._on_connected_callbacks.append(callback)
        return callback
    
    def on_disconnected(self, callback: Callable) -> Callable:
        """
        註冊斷開連接回調
        
        可作為裝飾器使用:
            @conn.on_disconnected
            def handle_disconnected():
                print("已斷開")
        """
        self._on_disconnected_callbacks.append(callback)
        return callback
    
    def on_error(self, callback: Callable) -> Callable:
        """
        註冊錯誤回調
        
        可作為裝飾器使用:
            @conn.on_error
            def handle_error(code, message):
                print(f"錯誤: {code} - {message}")
        """
        self._on_error_callbacks.append(callback)
        return callback
    
    # ========== 便捷方法 ==========
    
    def qualify_contracts(self, *contracts: Contract) -> List[Contract]:
        """
        驗證並取得完整合約資訊（同步）
        
        Args:
            contracts: 要驗證的合約
            
        Returns:
            驗證後的合約列表
        """
        return self._ib.qualifyContracts(*contracts)
    
    async def qualify_contracts_async(self, *contracts: Contract) -> List[Contract]:
        """
        驗證並取得完整合約資訊（異步）
        
        Args:
            contracts: 要驗證的合約
            
        Returns:
            驗證後的合約列表
        """
        return await self._ib.qualifyContractsAsync(*contracts)
    
    def get_account_values(self) -> List:
        """取得帳戶值"""
        return self._ib.accountValues()
    
    def get_portfolio(self) -> List:
        """取得持倉"""
        return self._ib.portfolio()
    
    def get_positions(self) -> List:
        """取得所有倉位"""
        return self._ib.positions()
    
    def get_open_orders(self) -> List[Order]:
        """取得未成交訂單"""
        return self._ib.openOrders()
    
    def get_open_trades(self) -> List[Trade]:
        """取得進行中的交易"""
        return self._ib.openTrades()
    
    # ========== Context Manager ==========
    
    async def __aenter__(self) -> "IBConnection":
        """異步 context manager 進入"""
        await self.connect_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """異步 context manager 退出"""
        await self.disconnect_async()


# ============================================================
# Context Manager 工廠函數
# ============================================================

@asynccontextmanager
async def ib_connection(
    config: Optional[ConnectionConfig] = None,
    event_bus: Optional[EventBus] = None,
):
    """
    IB 連接的異步 context manager
    
    使用方式:
        async with ib_connection() as conn:
            contracts = conn.qualify_contracts(stock)
            # ... 使用連接
    
    Args:
        config: 連接配置
        event_bus: 事件總線
        
    Yields:
        IBConnection 實例
    """
    conn = IBConnection(config=config, event_bus=event_bus)
    
    try:
        await conn.connect_async()
        yield conn
    finally:
        await conn.disconnect_async()


# ============================================================
# 便捷函數
# ============================================================

def create_connection_config(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    readonly: bool = False,
    timeout: int = 30,
    auto_reconnect: bool = True,
    max_reconnect_attempts: int = 5,
    reconnect_interval: int = 10,
) -> ConnectionConfig:
    """建立連接配置的便捷函數"""
    return ConnectionConfig(
        host=host,
        port=port,
        client_id=client_id,
        readonly=readonly,
        timeout=timeout,
        auto_reconnect=auto_reconnect,
        max_reconnect_attempts=max_reconnect_attempts,
        reconnect_interval=reconnect_interval,
    )