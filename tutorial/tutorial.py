import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List, TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)

class KelpStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.last_price = None  # Store the last mid-price for mean reversion

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return self.last_price if self.last_price is not None else 2_020 # Default fair value

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # filter offers with small volume
        adverse_volume = 12 
        filtered_ask = [price for price in order_depth.sell_orders if abs(order_depth.sell_orders[price]) >= adverse_volume]
        filtered_bid = [price for price in order_depth.buy_orders if abs(order_depth.buy_orders[price]) >= adverse_volume]

        mm_ask = min(filtered_ask) if filtered_ask else None
        mm_bid = max(filtered_bid) if filtered_bid else None

        # compute fair
        if mm_ask is None or mm_bid is None:
            mid_price = (best_ask + best_bid) / 2
        else:
            mid_price = (mm_ask + mm_bid) / 2

        # predict using last round price
        reversion_beta = 0.1  # Tune this parameter
        if self.last_price is not None:
            last_returns = (mid_price - self.last_price) / self.last_price
            pred_returns = last_returns * reversion_beta
            fair_value = mid_price + (mid_price * pred_returns)
        else:
            fair_value = mid_price

        self.last_price = mid_price  # Update last price for next iteration
        return int(fair_value+0.1)

class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000


class Trader:
    def __init__(self) -> None:
        limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50  # You can update the limit for other symbols as needed
        }

        # Assign strategies: KelpStrategy for "KELP", OtherStrategy for everything else
        self.strategies = {
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "SQUID_INK": RainforestResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
        }

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Processes market orders and determines appropriate actions per strategy.
        Uses KelpStrategy for KELP and OtherStrategy for all other symbols.
        """
        logger.print(state.position)

        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders = {}
        for symbol in state.order_depths:
            # if symbol not in self.strategies:
            #     # Assign OtherStrategy dynamically for non-KELP symbols
            #     self.strategies[symbol] = OtherStrategy(symbol, 10)  # Update limit if needed

            strategy = self.strategies[symbol]

            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))

            orders[symbol] = strategy.run(state)
            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
