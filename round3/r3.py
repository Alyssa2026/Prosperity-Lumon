import json
from abc import abstractmethod
from collections import deque
import statistics

import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List, TypeAlias
from statistics import NormalDist
import math

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
        # raise NotImplementedError()
        pass

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
        # added more checking
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return
        
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

        max_buy_price = true_value - 1 if position > self.limit * 0.4 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.4 else true_value

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

    def get_true_value(self, state: TradingState) -> int:
        # added empty check
        order_depth = state.order_depths[self.symbol]
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)
        

class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000

class SquidInkStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.buy_z = .4
        self.sell_z = .4
        self.mean = 1924.95084375
        self.std = 70.23916397594667
        self.max_z = 3 # used for scaling aggressiveness

    def get_mid_price(self, state: TradingState, symbol: str) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        return (buy_orders[0][0] + sell_orders[0][0]) / 2

    def act(self, state: TradingState) -> None:
        mid_price = self.get_mid_price(state, self.symbol)
        if mid_price is None:
            return

        diff = mid_price - self.mean
        z = (diff) / self.std if self.std != 0 else 0

        if z < -self.buy_z:
            self.go_long(state, z)
        elif z > self.sell_z:
            self.go_short(state, z)

    def scale_quantity(self, z: float) -> int:
        # Cap the z-score to avoid over-scaling
        abs_z = min(abs(z), self.max_z)
        # Linear scaling between 0 and 1
        scale = abs_z / self.max_z #
        return int(self.limit * scale)

    def go_long(self, state: TradingState, z: float) -> None:
        order_depth = state.order_depths[self.symbol]
        if not order_depth.sell_orders:
            return
        price = max(order_depth.sell_orders.keys())
        position = state.position.get(self.symbol, 0)
        newLimit = self.scale_quantity(z) # 1/3* limit
        quantity = min(newLimit-position, self.limit - position)
        if quantity > 0:
            self.buy(price, quantity)
            
    def go_short(self, state: TradingState, z: float) -> None:
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders:
            return
        price = min(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        newLimit = self.scale_quantity(z) # 1/3* limit

        quantity = min(newLimit+position, self.limit + position)
        if quantity > 0:
            self.sell(price, quantity)
    

class CombinedBasketStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, components: dict[str, int],
                 buy_z: float, sell_z: float, exit_z: float, mean: float, std: float) -> None:
        super().__init__(symbol, limit)
        self.components = components
        self.buy_z = buy_z
        self.sell_z = sell_z
        self.exit_z = exit_z
        self.mean = mean
        self.std = std

    def get_mid_price(self, state: TradingState, symbol: str) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        return (buy_orders[0][0] + sell_orders[0][0]) / 2

    def act(self, state: TradingState) -> None:
        mid_prices = {self.symbol: self.get_mid_price(state, self.symbol)}
        for symbol in self.components:
            mid_prices[symbol] = self.get_mid_price(state, symbol)

        if any(p is None for p in mid_prices.values()):
            return

        fair_value = sum(qty * mid_prices[symbol] for symbol, qty in self.components.items())
        market_price = mid_prices[self.symbol]
        diff = market_price - fair_value
        z = (diff - self.mean) / self.std if self.std != 0 else 0
        position = state.position.get(self.symbol, 0)

        if z < -self.buy_z and position < self.limit:
            self.go_long(state)
        elif z > self.sell_z and position > -self.limit:
            self.go_short(state)
        elif abs(z) < self.exit_z and position != 0:
            self.close_position(state, position)

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        self.buy(price, to_buy)


    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position
        self.sell(price, to_sell)

    def close_position(self, state: TradingState, position: int) -> None:
        order_depth = state.order_depths[self.symbol]
        if position > 0:
            # Sell to flatten
            price = min(order_depth.buy_orders.keys())
            self.sell(price, position)
        elif position < 0:
            # Buy to flatten
            price = max(order_depth.sell_orders.keys())
            self.buy(price, -position)


# This is the new pairs market making strategy.
class PairsMarketMakingStrategy(Strategy):
    def __init__(self, symbol1: str, symbol2: str, limit1: int, limit2: int,
                buy_threshold: float, sell_threshold: float,
                 exit_threshold: float, mean: float, std: float, order_size: int) -> None:
        # Call the parent constructor with one of the symbols for reference.
        super().__init__(symbol1, limit1)
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.limit1 = limit1
        self.limit2 = limit2
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.exit_threshold = exit_threshold
        self.mean = mean
        self.std = std
        self.order_size = order_size
        # For the pair, orders will be stored in a dictionary mapping symbol -> list[Order]
        self.orders = {}

    def get_mid_price(self, state: TradingState, symbol: str) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    # Overridden buy/sell methods that accept a symbol
    def buy(self, symbol: str, price: int, quantity: int) -> None:
        if symbol not in self.orders:
            self.orders[symbol] = []
        self.orders[symbol].append(Order(symbol, price, quantity))

    def sell(self, symbol: str, price: int, quantity: int) -> None:
        if symbol not in self.orders:
            self.orders[symbol] = []
        self.orders[symbol].append(Order(symbol, price, -quantity))

    def act(self, state: TradingState) -> None:
        mid1 = self.get_mid_price(state, self.symbol1)
        mid2 = self.get_mid_price(state, self.symbol2)
        if mid1 is None or mid2 is None:
            return

        current_ratio = mid1 / mid2
        z = (current_ratio - self.mean) / self.std if self.std != 0 else 0

        pos1 = state.position.get(self.symbol1, 0)
        pos2 = state.position.get(self.symbol2, 0)

        # Cap-aware order size calculation
        def cap_order(pos: int, limit: int, desired: int, side: str) -> int:
            if side == "buy":
                return max(0, min(desired, limit - pos))
            elif side == "sell":
                return max(0, min(desired, limit + pos))
            return 0

        if z > self.sell_threshold:
            # Overvalued: sell symbol1, buy symbol2
            if self.symbol1 in state.order_depths and self.symbol2 in state.order_depths:
                depth1 = state.order_depths[self.symbol1]
                depth2 = state.order_depths[self.symbol2]

                if depth1.buy_orders and depth2.sell_orders:
                    price1 = max(depth1.buy_orders)
                    price2 = min(depth2.sell_orders)

                    qty1 = cap_order(pos1, self.limit1, self.order_size, "sell")
                    qty2 = cap_order(pos2, self.limit2, self.order_size, "buy")
                
                    self.sell(self.symbol1, price1, qty1)
                    self.buy(self.symbol2, price2, qty2)

        elif z < -self.buy_threshold:
            # Undervalued: buy symbol1, sell symbol2
            if self.symbol1 in state.order_depths and self.symbol2 in state.order_depths:
                depth1 = state.order_depths[self.symbol1]
                depth2 = state.order_depths[self.symbol2]

                if depth1.sell_orders and depth2.buy_orders:
                    price1 = min(depth1.sell_orders)
                    price2 = max(depth2.buy_orders)

                    qty1 = cap_order(pos1, self.limit1, self.order_size, "buy")
                    qty2 = cap_order(pos2, self.limit2, self.order_size, "sell")

                   
                    self.buy(self.symbol1, price1, qty1)
                    self.sell(self.symbol2, price2, qty2)

        elif abs(z) < self.exit_threshold:
            # Mean reversion: close positions
            depth1 = state.order_depths.get(self.symbol1)
            depth2 = state.order_depths.get(self.symbol2)

            if pos1 > 0 and depth1 and depth1.buy_orders:
                price1 = max(depth1.buy_orders)
                self.sell(self.symbol1, price1, pos1)
            elif pos1 < 0 and depth1 and depth1.sell_orders:
                price1 = min(depth1.sell_orders)
                self.buy(self.symbol1, price1, -pos1)

            if pos2 > 0 and depth2 and depth2.buy_orders:
                price2 = max(depth2.buy_orders)
                self.sell(self.symbol2, price2, pos2)
            elif pos2 < 0 and depth2 and depth2.sell_orders:
                price2 = min(depth2.sell_orders)
                self.buy(self.symbol2, price2, -pos2)

    # We override run() so that it returns a dictionary mapping each symbol to its list of orders.
    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.orders = {}
        self.act(state)
        return self.orders

    def save(self) -> JSON:
        return {}

    def load(self, data: JSON) -> None:
        pass
    
class DjembeRatioArbitrageStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.symbol_pb = "PICNIC_BASKET1"
        self.mean_ratio = 0.227537
        self.std_ratio = 0.000907
        self.entry_z = 0.48
        self.exit_z = 0.133

    def get_mid_price(self, state: TradingState, symbol: str) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        return (best_bid + best_ask) / 2

    def act(self, state: TradingState) -> None:
        dj_price = self.get_mid_price(state, self.symbol)
        pb_price = self.get_mid_price(state, self.symbol_pb)
        if dj_price is None or pb_price is None:
            return

        ratio = dj_price / pb_price
        z = (ratio - self.mean_ratio) / self.std_ratio if self.std_ratio != 0 else 0

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]
         # added more checking
        if not order_depth:
            return

        if z > self.entry_z and position > -self.limit:
            # DJEMBES too expensive: Sell
            to_sell = self.limit + position
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                qty = min(volume, to_sell)
                self.sell(price, qty)
                to_sell -= qty
                if to_sell <= 0:
                    break

        elif z < -self.entry_z and position < self.limit:
            # DJEMBES too cheap: Buy
            to_buy = self.limit - position
            for price, volume in sorted(order_depth.sell_orders.items()):
                qty = min(-volume, to_buy)
                self.buy(price, qty)
                to_buy -= qty
                if to_buy <= 0:
                    break

        elif abs(z) < self.exit_z and position != 0:
            # Exit trade
            if position > 0:
                for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    qty = min(volume, position)
                    self.sell(price, qty)
                    position -= qty
                    if position <= 0:
                        break
            elif position < 0:
                for price, volume in sorted(order_depth.sell_orders.items()):
                    qty = min(-volume, -position)
                    self.buy(price, qty)
                    position += qty
                    if position >= 0:
                        break
def BS_CALL(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Call Option Pricing"""
    if T <= 0 or sigma == 0:
        return max(0, S - K)
    N = NormalDist().cdf
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * N(d1) - K * math.exp(-r*T) * N(d2)

EXPIRY_DAY = 7
CURRENT_ROUND = 3

from math import log, sqrt, exp
from statistics import NormalDist

def BS_CALL(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * NormalDist().cdf(d1) - K * exp(-r * T) * NormalDist().cdf(d2)

def implied_volatility(S, K, T, r, market_price, tol=1e-5, max_iter=100):
    # Use binary search for IV between 0.01 and 3.0
    low, high = 0.01, 3.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = BS_CALL(S, K, T, r, mid)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            high = mid
        else:
            low = mid
    return mid


from collections import deque

class VolcanicVoucherStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, strike_price: int) -> None:
        super().__init__(symbol, limit)
        self.strike_price = strike_price
        self.r = 0
        self.threshold = 2
        self.vol_window = deque(maxlen=25)  # rolling window of IVs
        self.prev_sigma = 0.17

    def get_mid_price(self, state: TradingState, symbol: str) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        bid = max(order_depth.buy_orders)
        ask = min(order_depth.sell_orders)
        return (bid + ask) / 2

    def act(self, state: TradingState) -> None:
        voucher_price = self.get_mid_price(state, self.symbol)
        rock_price = self.get_mid_price(state, "VOLCANIC_ROCK")
        if voucher_price is None or rock_price is None:
            return

        fractional_day = CURRENT_ROUND + (state.timestamp / 1_000_000)
        days_to_expiry = max(0.0, EXPIRY_DAY - fractional_day)
        T = days_to_expiry / 365

        # === Step 1: Price current voucher using average volatility
        fair_value = BS_CALL(S=rock_price, K=self.strike_price, T=T, r=self.r, sigma=self.prev_sigma)
        logger.print("FAIR VALUE: ", round(fair_value))
        logger.print("MARKET VALUE: ", round(voucher_price))
        logger.print("PERFECTLY PRICED" if abs(round(fair_value) - round(voucher_price)) <= 2 else "MISPRICED")

        # === Step 2: Trade on mispricing
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if voucher_price > fair_value + self.threshold:
            to_sell = self.limit + position
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if price >= fair_value + self.threshold:
                    qty = min(volume, to_sell)
                    self.sell(price, qty)
                    to_sell -= qty
                    if to_sell <= 0:
                        break
        elif voucher_price < fair_value - self.threshold:
            to_buy = self.limit - position
            for price, volume in sorted(order_depth.sell_orders.items()):
                if price <= fair_value - self.threshold:
                    qty = min(-volume, to_buy)
                    self.buy(price, qty)
                    to_buy -= qty
                    if to_buy <= 0:
                        break

        # === Step 3: Update implied volatility estimate from market price
        iv = implied_volatility(S=rock_price, K=self.strike_price, T=T, r=self.r, market_price=voucher_price)
        self.vol_window.append(iv)
        self.prev_sigma = sum(self.vol_window) / len(self.vol_window)

    def save(self) -> JSON:
        return {
            "prev_sigma": self.prev_sigma,
            "vol_window": list(self.vol_window),
        }

    def load(self, data: JSON) -> None:
        if data:
            self.prev_sigma = data.get("prev_sigma", 0.17)
            self.vol_window = deque(data.get("vol_window", []), maxlen=10)



                
# Modified Trader that integrates the pairs strategy.
class Trader:
    def __init__(self) -> None:
        # limits = {
        #     "KELP": 50,
        #     "RAINFOREST_RESIN": 50,
        #     "SQUID_INK": 50,
        #     "CROISSANTS": 250,
        #     "JAMS": 350,
        #     "DJEMBES": 60,
        #     "PICNIC_BASKET1": 60,
        #     "PICNIC_BASKET2": 100,
        #     "VOLCANIC_ROCK": 400,
        #     "VOLCANIC_ROCK_VOUCHER_9500": 200,
        #     "VOLCANIC_ROCK_VOUCHER_9750": 200,
        #     "VOLCANIC_ROCK_VOUCHER_10000": 200,
        #     "VOLCANIC_ROCK_VOUCHER_10250": 200,
        #     "VOLCANIC_ROCK_VOUCHER_10500": 200
        # }
        limits = {
            "KELP": 0,
            "RAINFOREST_RESIN": 0,
            "SQUID_INK": 0,
            "CROISSANTS": 0,
            "JAMS": 0,
            "DJEMBES": 0,
            "PICNIC_BASKET1": 0,
            "PICNIC_BASKET2": 0,
            "VOLCANIC_ROCK": 0,
            "VOLCANIC_ROCK_VOUCHER_9500": 0,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 0,
            "VOLCANIC_ROCK_VOUCHER_10250": 0,
            "VOLCANIC_ROCK_VOUCHER_10500": 0,
        }
        
        # Store individual strategies and, for croissants and jam, a combined pairs strategy.
        # Note: Remove the separate "CROISSANTS" and "JAMS" strategies since they are handled as a pair.
        self.strategies = {
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "SQUID_INK": SquidInkStrategy("SQUID_INK", limits["SQUID_INK"]),
            "DJEMBES": DjembeRatioArbitrageStrategy("DJEMBES", limits["DJEMBES"]),
            "PICNIC_BASKET1": CombinedBasketStrategy(
                "PICNIC_BASKET1", limits["PICNIC_BASKET1"],
                {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                buy_z=0.7, sell_z=0.7, exit_z=0.2,
                mean=48.76, std=85.91
            ),
            "PICNIC_BASKET2": CombinedBasketStrategy(
                "PICNIC_BASKET2", limits["PICNIC_BASKET2"],
                {"CROISSANTS": 4, "JAMS": 2},
                buy_z=2.2, sell_z=0.9, exit_z=0.1,
                mean=30.24, std=59.85
            ),
          "CROISSANTS_JAMS": PairsMarketMakingStrategy(
            "CROISSANTS", "JAMS",
            limits["CROISSANTS"],
            limits["JAMS"],
            buy_threshold=.6,        # Increase threshold if data indicates a larger move is required.
            sell_threshold=1,       # Likewise for the sell threshold.
            exit_threshold=0.15,      # Adjust exit threshold to match reversion characteristics.
            mean=0.6519  ,          # Update to reflect the historical mean ratio.
            std=0.0032  ,       # Update to reflect the measured standard deviation.
            order_size=20         # Adjust order size based on trade frequency and liquidity.
        ), 
        

        }

    # Add vouchers separately with initial expiry of 7
        #for strike in [9500, 9750, 10000, 10250, 10500]:
        for strike in [9750]:
            symbol = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            self.strategies[symbol] = VolcanicVoucherStrategy(symbol, limits[symbol], strike_price=strike)

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        logger.print(state.position)
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        orders: Dict[str, List[Order]] = {}

        for key, strategy in self.strategies.items():
            if key in old_trader_data:
                strategy.load(old_trader_data[key])
            result = strategy.run(state)
            if isinstance(result, list):  # single-symbol strategy
                orders[strategy.symbol] = result
            elif isinstance(result, dict):  # multi-symbol strategy
                orders.update(result)

            new_trader_data[key] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data