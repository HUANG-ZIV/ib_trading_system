"""
Registry 模組 - 策略註冊表

管理策略類別的註冊和策略實例的建立
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Type, Any, Callable
import threading

from .base import BaseStrategy, StrategyState, StrategyConfig


# 設定 logger
logger = logging.getLogger(__name__)


@dataclass
class StrategyClassInfo:
    """策略類別資訊"""
    
    name: str
    strategy_class: Type[BaseStrategy]
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    
    # 預設配置
    default_config: Optional[StrategyConfig] = None
    
    # 元數據
    tags: List[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyInstanceInfo:
    """策略實例資訊"""
    
    instance_id: str
    strategy: BaseStrategy
    class_name: str
    
    # 時間戳
    created_at: datetime = field(default_factory=datetime.now)


class StrategyRegistry:
    """
    策略註冊表
    
    管理策略類別的註冊和策略實例的建立
    
    使用方式:
        registry = StrategyRegistry()
        
        # 註冊策略類別
        registry.register_class("ma_cross", MACrossStrategy)
        
        # 建立策略實例
        strategy = registry.create("ma_cross", symbols=["AAPL"])
        
        # 使用裝飾器註冊
        @registry.register("rsi_strategy")
        class RSIStrategy(BaseStrategy):
            pass
    """
    
    def __init__(self):
        """初始化註冊表"""
        # 策略類別: {class_name: StrategyClassInfo}
        self._classes: Dict[str, StrategyClassInfo] = {}
        
        # 策略實例: {instance_id: StrategyInstanceInfo}
        self._instances: Dict[str, StrategyInstanceInfo] = {}
        
        # 線程安全
        self._lock = threading.RLock()
        
        logger.debug("StrategyRegistry 初始化完成")
    
    # ========== 類別註冊 ==========
    
    def register_class(
        self,
        name: str,
        strategy_class: Type[BaseStrategy],
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        default_config: Optional[StrategyConfig] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        註冊策略類別
        
        Args:
            name: 策略名稱（唯一識別符）
            strategy_class: 策略類別
            description: 策略描述
            version: 版本號
            author: 作者
            default_config: 預設配置
            tags: 標籤列表
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"{strategy_class} 必須繼承 BaseStrategy")
        
        with self._lock:
            if name in self._classes:
                logger.warning(f"策略類別 {name} 已存在，將被覆蓋")
            
            self._classes[name] = StrategyClassInfo(
                name=name,
                strategy_class=strategy_class,
                description=description,
                version=version,
                author=author,
                default_config=default_config,
                tags=tags or [],
            )
            
            logger.info(f"註冊策略類別: {name} ({strategy_class.__name__})")
    
    def register(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> Callable[[Type[BaseStrategy]], Type[BaseStrategy]]:
        """
        策略類別註冊裝飾器
        
        使用方式:
            @registry.register("my_strategy")
            class MyStrategy(BaseStrategy):
                pass
        """
        def decorator(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
            self.register_class(
                name=name,
                strategy_class=cls,
                description=description,
                version=version,
                author=author,
                tags=tags,
            )
            return cls
        return decorator
    
    def unregister_class(self, name: str) -> bool:
        """
        取消註冊策略類別
        
        Args:
            name: 策略名稱
            
        Returns:
            是否成功取消
        """
        with self._lock:
            if name in self._classes:
                del self._classes[name]
                logger.info(f"取消註冊策略類別: {name}")
                return True
            return False
    
    def get_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """取得策略類別"""
        info = self._classes.get(name)
        return info.strategy_class if info else None
    
    def get_class_info(self, name: str) -> Optional[StrategyClassInfo]:
        """取得策略類別資訊"""
        return self._classes.get(name)
    
    def get_all_classes(self) -> Dict[str, StrategyClassInfo]:
        """取得所有策略類別"""
        return self._classes.copy()
    
    def list_classes(self) -> List[str]:
        """列出所有策略類別名稱"""
        return list(self._classes.keys())
    
    def has_class(self, name: str) -> bool:
        """是否有指定策略類別"""
        return name in self._classes
    
    # ========== 實例建立 ==========
    
    def create(
        self,
        class_name: str,
        instance_id: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        config: Optional[StrategyConfig] = None,
        **kwargs,
    ) -> Optional[BaseStrategy]:
        """
        建立策略實例
        
        Args:
            class_name: 策略類別名稱
            instance_id: 實例 ID，None 則自動生成
            symbols: 訂閱標的
            config: 策略配置
            **kwargs: 傳遞給策略建構函數的額外參數
            
        Returns:
            策略實例或 None
        """
        class_info = self._classes.get(class_name)
        if class_info is None:
            logger.error(f"策略類別 {class_name} 未註冊")
            return None
        
        try:
            # 使用預設配置或提供的配置
            if config is None and class_info.default_config:
                config = class_info.default_config
            
            # 建立實例
            strategy = class_info.strategy_class(
                strategy_id=instance_id,
                symbols=symbols,
                config=config,
                **kwargs,
            )
            
            # 取得實際的 instance_id
            actual_id = strategy.strategy_id
            
            # 註冊實例
            with self._lock:
                self._instances[actual_id] = StrategyInstanceInfo(
                    instance_id=actual_id,
                    strategy=strategy,
                    class_name=class_name,
                )
            
            logger.info(f"建立策略實例: {actual_id} (class={class_name})")
            return strategy
            
        except Exception as e:
            logger.error(f"建立策略實例失敗 (class={class_name}): {e}")
            return None
    
    # ========== 實例管理 ==========
    
    def add(self, strategy: BaseStrategy, class_name: str = "") -> str:
        """
        添加已建立的策略實例
        
        Args:
            strategy: 策略實例
            class_name: 策略類別名稱（用於記錄）
            
        Returns:
            實例 ID
        """
        instance_id = strategy.strategy_id
        
        with self._lock:
            self._instances[instance_id] = StrategyInstanceInfo(
                instance_id=instance_id,
                strategy=strategy,
                class_name=class_name or strategy.__class__.__name__,
            )
        
        logger.info(f"添加策略實例: {instance_id}")
        return instance_id
    
    def get(self, instance_id: str) -> Optional[BaseStrategy]:
        """取得策略實例"""
        info = self._instances.get(instance_id)
        return info.strategy if info else None
    
    def get_info(self, instance_id: str) -> Optional[StrategyInstanceInfo]:
        """取得策略實例資訊"""
        return self._instances.get(instance_id)
    
    def remove(self, instance_id: str) -> bool:
        """
        移除策略實例
        
        Args:
            instance_id: 實例 ID
            
        Returns:
            是否成功移除
        """
        with self._lock:
            info = self._instances.get(instance_id)
            if info is None:
                return False
            
            # 停止策略
            if info.strategy.state == StrategyState.RUNNING:
                info.strategy.stop()
            
            # 清理資源
            info.strategy.cleanup()
            
            del self._instances[instance_id]
            logger.info(f"移除策略實例: {instance_id}")
            return True
    
    def remove_all(self) -> int:
        """
        移除所有策略實例
        
        Returns:
            移除的數量
        """
        with self._lock:
            count = len(self._instances)
            instance_ids = list(self._instances.keys())
        
        for instance_id in instance_ids:
            self.remove(instance_id)
        
        return count
    
    # ========== 查詢方法 ==========
    
    def get_all(self) -> Dict[str, BaseStrategy]:
        """取得所有策略實例"""
        return {
            instance_id: info.strategy
            for instance_id, info in self._instances.items()
        }
    
    def get_active(self) -> Dict[str, BaseStrategy]:
        """取得所有運行中的策略"""
        return {
            instance_id: info.strategy
            for instance_id, info in self._instances.items()
            if info.strategy.state == StrategyState.RUNNING
        }
    
    def get_by_class(self, class_name: str) -> List[BaseStrategy]:
        """取得指定類別的所有實例"""
        return [
            info.strategy
            for info in self._instances.values()
            if info.class_name == class_name
        ]
    
    def get_by_symbol(self, symbol: str) -> List[BaseStrategy]:
        """取得訂閱指定標的的所有策略"""
        return [
            info.strategy
            for info in self._instances.values()
            if symbol in info.strategy.symbols
        ]
    
    def get_by_state(self, state: StrategyState) -> List[BaseStrategy]:
        """取得指定狀態的所有策略"""
        return [
            info.strategy
            for info in self._instances.values()
            if info.strategy.state == state
        ]
    
    def list_instances(self) -> List[str]:
        """列出所有實例 ID"""
        return list(self._instances.keys())
    
    def has_instance(self, instance_id: str) -> bool:
        """是否有指定實例"""
        return instance_id in self._instances
    
    # ========== 統計 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """取得註冊表統計"""
        with self._lock:
            state_counts = {}
            for info in self._instances.values():
                state = info.strategy.state.name
                state_counts[state] = state_counts.get(state, 0) + 1
            
            return {
                "registered_classes": len(self._classes),
                "total_instances": len(self._instances),
                "state_counts": state_counts,
                "classes": list(self._classes.keys()),
                "instances": list(self._instances.keys()),
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """取得詳細摘要"""
        with self._lock:
            classes_info = {
                name: {
                    "description": info.description,
                    "version": info.version,
                    "author": info.author,
                    "tags": info.tags,
                }
                for name, info in self._classes.items()
            }
            
            instances_info = {
                instance_id: {
                    "class_name": info.class_name,
                    "state": info.strategy.state.name,
                    "symbols": list(info.strategy.symbols),
                    "created_at": info.created_at.isoformat(),
                }
                for instance_id, info in self._instances.items()
            }
            
            return {
                "classes": classes_info,
                "instances": instances_info,
            }


# ============================================================
# 全局單例
# ============================================================

_registry: Optional[StrategyRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> StrategyRegistry:
    """
    取得全局 StrategyRegistry 實例（單例模式）
    
    Returns:
        StrategyRegistry 實例
    """
    global _registry
    
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = StrategyRegistry()
    
    return _registry


def reset_registry() -> None:
    """重置全局註冊表（用於測試）"""
    global _registry
    
    with _registry_lock:
        if _registry is not None:
            _registry.remove_all()
        _registry = None


# ============================================================
# 便捷函數
# ============================================================

def register_strategy(
    name: str,
    description: str = "",
    version: str = "1.0.0",
    author: str = "",
    tags: Optional[List[str]] = None,
) -> Callable[[Type[BaseStrategy]], Type[BaseStrategy]]:
    """
    策略註冊裝飾器（使用全局註冊表）
    
    使用方式:
        @register_strategy("my_strategy", description="我的策略")
        class MyStrategy(BaseStrategy):
            pass
    """
    registry = get_registry()
    return registry.register(
        name=name,
        description=description,
        version=version,
        author=author,
        tags=tags,
    )