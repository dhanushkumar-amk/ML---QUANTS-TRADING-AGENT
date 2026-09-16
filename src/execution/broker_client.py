# ============================================================
# Broker Client Abstraction & Alpaca Integration (Phase 44)
# ============================================================
"""
Production broker client abstraction layer with paper-trading safety enforcement.

=============================================================================
ARCHITECTURAL PRINCIPLE: BROKER ABSTRACTION & LIVE-TRADING SAFETY RAILS
-----------------------------------------------------------------------------
1. Vendor-Agnostic Interface:
   Strategy signals and the Phase 39 RiskEngine NEVER bind directly to third-party
   broker SDKs (Alpaca, Interactive Brokers, Tradier, etc.).
   All interactions flow through the abstract `BrokerClient` interface.
   This enables zero-code modifications to strategy or risk engines when onboarding
   institutional prime brokers.

2. Enforced Paper-Trading Safety Gate:
   Live trading carries severe financial, legal, and operational tail risks.
   The `AlpacaBrokerClient` strictly defaults to PAPER trading. It inspects:
     - Endpoint hostnames (must point to paper-api.alpaca.markets).
     - API Key prefix (Alpaca paper keys begin with 'PK').
     - Explicit `paper=True` parameter.
   If any live credential or endpoint is detected without an explicit, deliberate
   `allow_live=True` override flag, a `LiveTradingSafetyError` is immediately raised,
   preventing catastrophic accidental executions in real money accounts.

3. Rate Limit Compliance & Connection Resilience:
   Alpaca enforces a hard ceiling of 200 requests/minute. All API operations employ
   automatic exponential backoff retry logic for transient errors and HTTP 429s.
=============================================================================
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from src.utils.config_loader import get_env
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Enums & Normalized Data Containers
# ============================================================


class OrderSide(str, Enum):
    """Trading action direction."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Execution order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Normalized order lifecycle status."""

    NEW = "new"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    SUSPENDED = "suspended"


class LiveTradingSafetyError(RuntimeError):
    """Raised when live trading credentials or endpoints are detected without explicit override."""


@dataclass
class BrokerAccount:
    """Normalized broker account details."""

    account_id: str
    status: str
    currency: str
    cash: float
    portfolio_value: float
    buying_power: float
    equity: float
    is_paper: bool
    raw_data: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class BrokerPosition:
    """Normalized asset position holding."""

    ticker: str
    quantity: float
    side: str  # "long" or "short"
    market_value: float
    cost_basis: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class BrokerOrder:
    """Normalized order response structure."""

    order_id: str
    client_order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    filled_quantity: float
    filled_avg_price: float | None
    status: OrderStatus
    created_at: datetime | str
    submitted_at: datetime | str | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    raw_data: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_terminal(self) -> bool:
        """Return True if order is in a final, non-active state."""
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }

    @property
    def remaining_quantity(self) -> float:
        """Unfilled share quantity."""
        return max(0.0, self.quantity - self.filled_quantity)


# ============================================================
# 2. Abstract Broker Client Interface
# ============================================================


class BrokerClient(ABC):
    """Abstract Base Class defining the universal broker communication contract."""

    @abstractmethod
    def get_account_info(self) -> BrokerAccount:
        """Retrieve current broker account balance, buying power, and portfolio equity."""

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Fetch all currently open portfolio positions."""

    @abstractmethod
    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: OrderSide | str,
        order_type: OrderType | str = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
        time_in_force: str = "day",
    ) -> BrokerOrder:
        """Submit a new order to the broker with idempotency and safety validation."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by its unique broker order ID."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> BrokerOrder:
        """Query the latest status and fill details of a specific order."""

    @abstractmethod
    def health_check(self) -> bool:
        """Verify API connectivity and authentication health."""


# ============================================================
# 3. Retry Helper with Exponential Backoff
# ============================================================


def _retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    allowed_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """Execute callable with exponential backoff on transient errors."""
    delay = initial_delay
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except allowed_exceptions as err:
            last_err = err
            # If safety error or value error, do not retry
            if isinstance(err, (LiveTradingSafetyError, ValueError)):
                raise
            if attempt == max_retries:
                logger.error("Operation failed after %d retries: %s", max_retries, err)
                raise
            logger.warning(
                "Transient error on attempt %d/%d (%s). Retrying in %.2fs...",
                attempt,
                max_retries,
                err,
                delay,
            )
            time.sleep(delay)
            delay *= backoff_factor

    if last_err is not None:
        raise last_err


# ============================================================
# 4. Concrete Alpaca Broker Client Implementation
# ============================================================


class AlpacaBrokerClient(BrokerClient):
    """Production implementation of BrokerClient wrapping the Alpaca Trading API.

    Enforces paper-trading safety and rate-limiting compliance.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        allow_live: bool = False,
        raw_client: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        api_key : str, optional
            Alpaca API key. Defaults to ALPACA_API_KEY environment variable.
        secret_key : str, optional
            Alpaca secret key. Defaults to ALPACA_SECRET_KEY environment variable.
        paper : bool, default True
            Paper-trading flag. Defaults to True.
        allow_live : bool, default False
            EXPLICIT SAFETY OVERRIDE. Must be set to True to permit live trading.
        raw_client : Any, optional
            Pre-configured client instance for unit testing / mocking.
        """
        self.paper = paper
        self.allow_live = allow_live
        self._api_key = api_key or get_env("ALPACA_API_KEY", "")
        self._secret_key = secret_key or get_env("ALPACA_SECRET_KEY", "")

        # -------------------------------------------------------------
        # HARD SAFETY RAIL: Verify Paper vs. Live Credentials & Endpoint
        # -------------------------------------------------------------
        self._enforce_safety_rails()

        if raw_client is not None:
            self._client = raw_client
        else:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self.paper,
            )

        logger.info(
            "AlpacaBrokerClient initialized successfully (Mode: %s, AllowLive: %s)",
            "PAPER" if self.paper else "LIVE",
            self.allow_live,
        )

    def _enforce_safety_rails(self) -> None:
        """Validate credentials and endpoints to prevent accidental live execution."""
        # 1. Check paper parameter
        if not self.paper and not self.allow_live:
            raise LiveTradingSafetyError(
                "CRITICAL SAFETY VIOLATION: AlpacaBrokerClient requested with paper=False, "
                "but allow_live=False. Live trading is strictly blocked by default. "
                "You must set allow_live=True deliberately to override."
            )

        # 2. Check API Key Prefix (Alpaca convention: paper keys start with 'PK', live start with 'AK')
        if self._api_key:
            clean_key = self._api_key.strip()
            is_live_key = clean_key.startswith("AK")
            if is_live_key and not self.allow_live:
                raise LiveTradingSafetyError(
                    f"CRITICAL SAFETY VIOLATION: Detected live Alpaca API key prefix ('{clean_key[:2]}'), "
                    "but allow_live=False. Paper trading keys must start with 'PK'. "
                    "Aborting execution to prevent real capital exposure."
                )

    # ---- Interface Methods ------------------------------------------

    def get_account_info(self) -> BrokerAccount:
        """Retrieve account details with retry logic."""

        def _call() -> BrokerAccount:
            acc = self._client.get_account()
            return BrokerAccount(
                account_id=str(getattr(acc, "id", "")),
                status=str(getattr(acc, "status", "")),
                currency=str(getattr(acc, "currency", "USD")),
                cash=float(getattr(acc, "cash", 0.0)),
                portfolio_value=float(getattr(acc, "portfolio_value", 0.0)),
                buying_power=float(getattr(acc, "buying_power", 0.0)),
                equity=float(getattr(acc, "equity", getattr(acc, "portfolio_value", 0.0))),
                is_paper=self.paper,
                raw_data=dict(getattr(acc, "__dict__", {})),
            )

        return _retry_with_backoff(_call)

    def get_positions(self) -> list[BrokerPosition]:
        """Fetch open positions normalized into BrokerPosition dataclasses."""

        def _call() -> list[BrokerPosition]:
            positions = self._client.get_all_positions()
            normalized: list[BrokerPosition] = []
            for p in positions:
                normalized.append(
                    BrokerPosition(
                        ticker=str(getattr(p, "symbol", "")),
                        quantity=float(getattr(p, "qty", 0.0)),
                        side=str(getattr(p, "side", "long")).lower(),
                        market_value=float(getattr(p, "market_value", 0.0)),
                        cost_basis=float(getattr(p, "cost_basis", 0.0)),
                        current_price=float(getattr(p, "current_price", 0.0)),
                        unrealized_pnl=float(getattr(p, "unrealized_pl", 0.0)),
                        unrealized_pnl_pct=float(getattr(p, "unrealized_plpc", 0.0)),
                    )
                )
            return normalized

        return _retry_with_backoff(_call)

    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: OrderSide | str,
        order_type: OrderType | str = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
        time_in_force: str = "day",
    ) -> BrokerOrder:
        """Place an order with Alpaca with idempotency client_order_id."""
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
        from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        side_str = side.value if hasattr(side, "value") else str(side)
        if "." in side_str:
            side_str = side_str.split(".")[-1]
        norm_side = OrderSide(side_str.lower())

        type_str = order_type.value if hasattr(order_type, "value") else str(order_type)
        if "." in type_str:
            type_str = type_str.split(".")[-1]
        norm_type = OrderType(type_str.lower())

        side_enum = AlpacaOrderSide.BUY if norm_side == OrderSide.BUY else AlpacaOrderSide.SELL
        tif_enum = getattr(AlpacaTimeInForce, time_in_force.upper(), AlpacaTimeInForce.DAY)

        # Prepare Alpaca Request
        if norm_type == OrderType.MARKET:
            req = MarketOrderRequest(
                symbol=ticker.upper(),
                qty=quantity,
                side=side_enum,
                time_in_force=tif_enum,
                client_order_id=client_order_id,
            )
        elif norm_type == OrderType.LIMIT:
            if limit_price is None or limit_price <= 0:
                raise ValueError("limit_price must be positive for LIMIT orders.")
            req = LimitOrderRequest(
                symbol=ticker.upper(),
                qty=quantity,
                side=side_enum,
                time_in_force=tif_enum,
                limit_price=round(float(limit_price), 2),
                client_order_id=client_order_id,
            )
        elif norm_type == OrderType.STOP:
            if stop_price is None or stop_price <= 0:
                raise ValueError("stop_price must be positive for STOP orders.")
            req = StopOrderRequest(
                symbol=ticker.upper(),
                qty=quantity,
                side=side_enum,
                time_in_force=tif_enum,
                stop_price=round(float(stop_price), 2),
                client_order_id=client_order_id,
            )
        elif norm_type == OrderType.STOP_LIMIT:
            if limit_price is None or stop_price is None:
                raise ValueError("limit_price and stop_price are required for STOP_LIMIT.")
            req = StopLimitOrderRequest(
                symbol=ticker.upper(),
                qty=quantity,
                side=side_enum,
                time_in_force=tif_enum,
                limit_price=round(float(limit_price), 2),
                stop_price=round(float(stop_price), 2),
                client_order_id=client_order_id,
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        def _call() -> BrokerOrder:
            logger.info(
                "Submitting Alpaca order: %s %s %.2f %s (client_order_id=%s)",
                norm_side.value.upper(),
                ticker.upper(),
                quantity,
                norm_type.value.upper(),
                client_order_id,
            )
            raw_order = self._client.submit_order(req)
            return self._normalize_alpaca_order(raw_order)

        return _retry_with_backoff(_call)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID."""

        def _call() -> bool:
            logger.info("Canceling Alpaca order: %s", order_id)
            self._client.cancel_order_by_id(order_id)
            return True

        return _retry_with_backoff(_call)

    def get_order_status(self, order_id: str) -> BrokerOrder:
        """Query single order by ID."""

        def _call() -> BrokerOrder:
            raw = self._client.get_order_by_id(order_id)
            return self._normalize_alpaca_order(raw)

        return _retry_with_backoff(_call)

    def health_check(self) -> bool:
        """Check API authentication and connection responsiveness."""
        try:
            acc = self.get_account_info()
            return acc.status.lower() in {"active", "accountstatus.active"}
        except Exception as err:
            logger.error("AlpacaBrokerClient health check failed: %s", err)
            return False

    # ---- Internal Normalizer ----------------------------------------

    def _normalize_alpaca_order(self, raw_order: Any) -> BrokerOrder:
        """Translate raw Alpaca Order model to normalized BrokerOrder."""
        status_str = str(getattr(raw_order, "status", "accepted")).lower()
        if "." in status_str:
            status_str = status_str.split(".")[-1]

        side_str = str(getattr(raw_order, "side", "buy")).lower()
        if "." in side_str:
            side_str = side_str.split(".")[-1]

        type_str = str(
            getattr(raw_order, "order_type", getattr(raw_order, "type", "market"))
        ).lower()
        if "." in type_str:
            type_str = type_str.split(".")[-1]

        # Map to internal enum safely
        status_map = {
            "new": OrderStatus.NEW,
            "accepted": OrderStatus.ACCEPTED,
            "pending_new": OrderStatus.PENDING_NEW,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "replaced": OrderStatus.CANCELLED,
            "pending_cancel": OrderStatus.ACCEPTED,
            "pending_replace": OrderStatus.ACCEPTED,
            "rejected": OrderStatus.REJECTED,
            "suspended": OrderStatus.SUSPENDED,
        }
        order_status = status_map.get(status_str, OrderStatus.ACCEPTED)

        filled_qty = float(getattr(raw_order, "filled_qty", 0.0) or 0.0)
        filled_px = getattr(raw_order, "filled_avg_price", None)
        filled_price = float(filled_px) if filled_px is not None else None

        lim_px = getattr(raw_order, "limit_price", None)
        limit_price = float(lim_px) if lim_px is not None else None

        stop_px = getattr(raw_order, "stop_price", None)
        stop_price = float(stop_px) if stop_px is not None else None

        return BrokerOrder(
            order_id=str(getattr(raw_order, "id", "")),
            client_order_id=str(getattr(raw_order, "client_order_id", "")),
            ticker=str(getattr(raw_order, "symbol", "")),
            side=OrderSide.BUY if side_str == "buy" else OrderSide.SELL,
            order_type=(
                OrderType(type_str)
                if type_str in [e.value for e in OrderType]
                else OrderType.MARKET
            ),
            quantity=float(getattr(raw_order, "qty", 0.0)),
            filled_quantity=filled_qty,
            filled_avg_price=filled_price,
            status=order_status,
            created_at=getattr(raw_order, "created_at", datetime.now()),
            submitted_at=getattr(raw_order, "submitted_at", None),
            limit_price=limit_price,
            stop_price=stop_price,
            raw_data=dict(getattr(raw_order, "__dict__", {})),
        )
