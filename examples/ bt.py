import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, Sequence


class ExitReason(Enum):
    TAKE_PROFIT = auto()
    SIGNAL_SELL = auto()
    END_OF_PERIOD = auto()


@dataclass
class Position:
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    take_profit_level: float | None = None


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    exit_reason: ExitReason


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    base_currency: pd.Series
    unrealized_pnl: pd.Series
    trades: list[Trade]


class TakeProfitStrategy(Protocol):
    def calculate_take_profit(
        self, entry_price: float, current_price: float
    ) -> float: ...

    def should_update_take_profit(
        self, position: Position, current_price: float
    ) -> bool: ...

    def update_take_profit(self, position: Position, current_price: float) -> float: ...


class StaticTakeProfit:
    def __init__(self, take_profit_pct: float):
        self.take_profit_pct = take_profit_pct

    def calculate_take_profit(self, entry_price: float, current_price: float) -> float:
        return entry_price * (1 + self.take_profit_pct)

    def should_update_take_profit(
        self, position: Position, current_price: float
    ) -> bool:
        return False

    def update_take_profit(self, position: Position, current_price: float) -> float:
        return (
            position.take_profit_level
            if position.take_profit_level
            else position.entry_price * (1 + self.take_profit_pct)
        )


class OrderManager(ABC):
    @abstractmethod
    def should_enter_position(
        self,
        t: pd.Timestamp,
        signals: pd.DataFrame,
        current_positions: Sequence[Position],
        last_position_time: pd.Timestamp | None,
    ) -> bool: ...

    @abstractmethod
    def should_exit_position(
        self,
        position: Position,
        current_price: float,
        signals: pd.DataFrame,
        t: pd.Timestamp,
    ) -> tuple[bool, ExitReason]: ...


class DefaultOrderManager(OrderManager):
    def __init__(
        self,
        max_positions: int,
        time_between_orders: pd.Timedelta,
        fix_loss_tol: float,
        tp_strategy: TakeProfitStrategy,
    ):
        self.max_positions = max_positions
        self.time_between_orders = time_between_orders
        self.fix_loss_tol = fix_loss_tol
        self.tp_strategy = tp_strategy

    def should_enter_position(
        self,
        t: pd.Timestamp,
        signals: pd.DataFrame,
        current_positions: Sequence[Position],
        last_position_time: pd.Timestamp | None,
    ) -> bool:
        if len(current_positions) >= self.max_positions:
            return False

        if last_position_time and (t - last_position_time) <= self.time_between_orders:
            return False

        return signals.loc[t, "signal"] == 1

    def should_exit_position(
        self,
        position: Position,
        current_price: float,
        signals: pd.DataFrame,
        t: pd.Timestamp,
    ) -> tuple[bool, ExitReason]:
        if position.take_profit_level and current_price >= position.take_profit_level:
            return True, ExitReason.TAKE_PROFIT

        if signals.loc[t, "signal_sell"] == 1:
            profit_pct = (current_price - position.entry_price) / position.entry_price
            if profit_pct < -self.fix_loss_tol:
                return True, ExitReason.SIGNAL_SELL

        return False, ExitReason.END_OF_PERIOD


class Backtester:
    def __init__(
        self,
        order_manager: OrderManager,
        initial_capital: float = 10000.0,
        position_size: float = 1.0,
    ):
        self.order_manager = order_manager
        self.initial_capital = initial_capital
        self.position_size = position_size

    def run(
        self,
        candles: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        pnl_series = pd.Series(0.0, index=signals.index)
        equity_curve = pd.Series(self.initial_capital, index=signals.index)
        base_currency = pd.Series(0.0, index=signals.index)
        unrealized_pnl = pd.Series(0.0, index=signals.index)

        trades: list[Trade] = []
        cash = self.initial_capital
        base_amount = 0.0
        positions: list[Position] = []
        last_position_time = None

        for t in signals.index:
            current_price = candles.loc[t, "close"]
            daily_realized_pnl = 0.0

            # Check and update take profits
            positions_to_close = []
            for pos in positions:
                should_exit, exit_reason = self.order_manager.should_exit_position(
                    pos, current_price, signals, t
                )

                if should_exit:
                    pnl = (current_price - pos.entry_price) * pos.quantity
                    daily_realized_pnl += pnl

                    cash += current_price * pos.quantity
                    base_amount -= pos.quantity

                    trades.append(
                        Trade(
                            entry_time=pos.entry_time,
                            exit_time=t,
                            entry_price=pos.entry_price,
                            exit_price=current_price,
                            quantity=pos.quantity,
                            pnl=pnl,
                            exit_reason=exit_reason,
                        )
                    )
                    positions_to_close.append(pos)

            for pos in positions_to_close:
                positions.remove(pos)

            # Check entry signals
            if self.order_manager.should_enter_position(
                t, signals, positions, last_position_time
            ):
                quantity = self.position_size
                cost = current_price * quantity

                if cost <= cash:
                    position = Position(
                        entry_time=t,
                        entry_price=current_price,
                        quantity=quantity,
                    )
                    if isinstance(self.order_manager, DefaultOrderManager):
                        position.take_profit_level = (
                            self.order_manager.tp_strategy.calculate_take_profit(
                                current_price, current_price
                            )
                        )

                    positions.append(position)
                    cash -= cost
                    base_amount += quantity
                    last_position_time = t

            # Calculate unrealized PnL
            current_unrealized = sum(
                (current_price - pos.entry_price) * pos.quantity for pos in positions
            )

            # Update series
            unrealized_pnl.loc[t] = current_unrealized
            if daily_realized_pnl != 0.0:
                pnl_series.loc[t] = daily_realized_pnl

            base_currency.loc[t] = base_amount
            total_equity = cash + (base_amount * current_price)
            equity_curve.loc[t] = total_equity

        # Close remaining positions
        if positions:
            final_price = candles.iloc[-1]["close"]
            final_realized_pnl = 0.0

            for pos in positions:
                pnl = (final_price - pos.entry_price) * pos.quantity
                final_realized_pnl += pnl

                cash += final_price * pos.quantity
                base_amount -= pos.quantity

                trades.append(
                    Trade(
                        entry_time=pos.entry_time,
                        exit_time=signals.index[-1],
                        entry_price=pos.entry_price,
                        exit_price=final_price,
                        quantity=pos.quantity,
                        pnl=pnl,
                        exit_reason=ExitReason.END_OF_PERIOD,
                    )
                )

            if final_realized_pnl != 0.0:
                pnl_series.loc[signals.index[-1]] = final_realized_pnl

            base_currency.loc[signals.index[-1]] = base_amount
            equity_curve.loc[signals.index[-1]] = cash + (base_amount * final_price)

        # Consistency check
        if trades:
            total_pnl_from_trades = sum(trade.pnl for trade in trades)
            total_pnl_from_series = pnl_series.sum()

            if not np.isclose(total_pnl_from_trades, total_pnl_from_series, rtol=1e-8):
                raise ValueError(
                    f"PnL mismatch: trades={total_pnl_from_trades:.2f} vs series={total_pnl_from_series:.2f}"
                )

        return BacktestResult(
            equity_curve=equity_curve,
            base_currency=base_currency,
            unrealized_pnl=unrealized_pnl,
            trades=trades,
        )


def backtest(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 10000.0,
    position_size: float = 1.0,
    take_profit: float = 0.05,
    max_positions: int = 3,
    fix_loss_tol: float = 0.01,
    time_between_two_consecutive_orders: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series, list[dict]]:
    tp_strategy = StaticTakeProfit(take_profit)
    order_manager = DefaultOrderManager(
        max_positions=max_positions,
        time_between_orders=pd.Timedelta(hours=time_between_two_consecutive_orders),
        fix_loss_tol=fix_loss_tol,
        tp_strategy=tp_strategy,
    )

    backtester = Backtester(
        order_manager=order_manager,
        initial_capital=initial_capital,
        position_size=position_size,
    )

    result = backtester.run(candles, signals)

    # Convert to old format for compatibility
    trades_dict = [
        {
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "quantity": trade.quantity,
            "exit_reason": trade.exit_reason.name.lower(),
        }
        for trade in result.trades
    ]

    return (
        result.equity_curve,
        result.base_currency,
        result.unrealized_pnl,
        trades_dict,
    )
