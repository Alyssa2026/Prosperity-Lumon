import json
from abc import abstractmethod
from math import log, sqrt
import math
from statistics import NormalDist
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from collections import deque
from typing import Any, Dict, List, TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility
    
    
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


class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    SYNTHETIC = "SYNTHETIC"

BASKET_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

class BasketTrader:
    def __init__(self, symbol: str, limit: int):
        self.symbol = symbol
        self.limit = limit
        self.spread_history = deque(maxlen=100)  # For dynamic thresholding


    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        synthetic_depth = OrderDepth()

        croissants_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys(), default=0)
        croissants_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys(), default=float("inf"))

        jams_bid = max(order_depths[Product.JAMS].buy_orders.keys(), default=0)
        jams_ask = min(order_depths[Product.JAMS].sell_orders.keys(), default=float("inf"))

        djembes_bid = max(order_depths[Product.DJEMBES].buy_orders.keys(), default=0)
        djembes_ask = min(order_depths[Product.DJEMBES].sell_orders.keys(), default=float("inf"))

        implied_bid = croissants_bid * BASKET_WEIGHTS[Product.CROISSANTS] + \
                       jams_bid * BASKET_WEIGHTS[Product.JAMS] + \
                       djembes_bid * BASKET_WEIGHTS[Product.DJEMBES]

        implied_ask = croissants_ask * BASKET_WEIGHTS[Product.CROISSANTS] + \
                       jams_ask * BASKET_WEIGHTS[Product.JAMS] + \
                       djembes_ask * BASKET_WEIGHTS[Product.DJEMBES]

        if implied_bid > 0:
            vol_bid = min(
                order_depths[Product.CROISSANTS].buy_orders.get(croissants_bid, 0) // BASKET_WEIGHTS[Product.CROISSANTS],
                order_depths[Product.JAMS].buy_orders.get(jams_bid, 0) // BASKET_WEIGHTS[Product.JAMS],
                order_depths[Product.DJEMBES].buy_orders.get(djembes_bid, 0) // BASKET_WEIGHTS[Product.DJEMBES],
            )
            synthetic_depth.buy_orders[implied_bid] = vol_bid

        if implied_ask < float("inf"):
            vol_ask = min(
                -order_depths[Product.CROISSANTS].sell_orders.get(croissants_ask, 0) // BASKET_WEIGHTS[Product.CROISSANTS],
                -order_depths[Product.JAMS].sell_orders.get(jams_ask, 0) // BASKET_WEIGHTS[Product.JAMS],
                -order_depths[Product.DJEMBES].sell_orders.get(djembes_ask, 0) // BASKET_WEIGHTS[Product.DJEMBES],
            )
            synthetic_depth.sell_orders[implied_ask] = -vol_ask
        logger.print("synth")
        logger.print(synthetic_depth.sell_orders)
        return synthetic_depth

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity

            if quantity > 0:
                croissant_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                jam_price = min(order_depths[Product.JAMS].sell_orders.keys())
                djembe_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            else:
                croissant_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jam_price = max(order_depths[Product.JAMS].buy_orders.keys())
                djembe_price = max(order_depths[Product.DJEMBES].buy_orders.keys())

            component_orders[Product.CROISSANTS].append(Order(
                Product.CROISSANTS, croissant_price, quantity * BASKET_WEIGHTS[Product.CROISSANTS]))
            component_orders[Product.JAMS].append(Order(
                Product.JAMS, jam_price, quantity * BASKET_WEIGHTS[Product.JAMS]))
            component_orders[Product.DJEMBES].append(Order(
                Product.DJEMBES, djembe_price, quantity * BASKET_WEIGHTS[Product.DJEMBES]))

        return component_orders

    def execute_spread_orders(self, basket_position: int, order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]] | None:
        basket_od = order_depths[Product.PICNIC_BASKET1]
        synthetic_od = self.get_synthetic_basket_order_depth(order_depths)

        try:
            basket_ask = min(basket_od.sell_orders)
            basket_bid = max(basket_od.buy_orders)
            synthetic_ask = min(synthetic_od.sell_orders)
            synthetic_bid = max(synthetic_od.buy_orders)
        except ValueError:
            return None  # Some order book is empty

        # --- Spread logic ---
        spread_buy = synthetic_bid - basket_ask  # We buy basket, sell synthetic
        spread_sell = basket_bid - synthetic_ask  # We sell basket, buy synthetic
        # Add spread to history
        if synthetic_bid and basket_ask:
            self.spread_history.append(synthetic_bid - basket_ask)

        # Default threshold fallback
        default_threshold = 10

        # Dynamic threshold using std deviation
        if len(self.spread_history) >= 10:
            spread_std = np.std(self.spread_history)
            threshold = max(10, spread_std *1.2)  # Don't go too low
        
        else:
            threshold = default_threshold
       

        # --- Maximum allowed trade size per opportunity ---
        max_size = 5  # hard cap for safety

        # --- If synthetic overpriced, sell synthetic, buy basket ---
        if spread_buy > threshold:
            confidence = min((spread_buy - threshold) / threshold, 1.0)
            volume = int(confidence * max_size)

            # Cap volume by order book depth
            volume = min(volume, abs(basket_od.sell_orders[basket_ask]), abs(synthetic_od.buy_orders[synthetic_bid]))

            if volume <= 0:
                return None

            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid, -volume)]
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_ask, volume)]
            result = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            result[Product.PICNIC_BASKET1] = basket_orders
            return result

        # --- If basket overpriced, sell basket, buy synthetic ---
        elif spread_sell > threshold:
            confidence = min((spread_sell - threshold) / threshold, 1.0)
            volume = int(confidence * max_size)

            volume = min(volume, abs(basket_od.buy_orders[basket_bid]), abs(synthetic_od.sell_orders[synthetic_ask]))

            if volume <= 0:
                return None

            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask, volume)]
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_bid, -volume)]
            result = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            result[Product.PICNIC_BASKET1] = basket_orders
            return result

        return None

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        basket_position = state.position.get(self.symbol, 0)
        result = self.execute_spread_orders(basket_position, state.order_depths)
        return result if result else {}
   
    def save(self) -> JSON:
        return {}

    def load(self, data: JSON) -> None:
        pass

class Product2:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    SYNTHETIC = "SYNTHETIC"

BASKET_WEIGHTS2 = {
    Product2.CROISSANTS: 4,
    Product2.JAMS: 2,
    Product2.DJEMBES: 0,
}

class BasketTrader2:
    def __init__(self, symbol: str, limit: int):
        self.symbol = symbol
        self.limit = limit
        self.spread_history = deque(maxlen=100)  # For dynamic thresholding
    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        synthetic_depth = OrderDepth()

        croissants_bid = max(order_depths[Product2.CROISSANTS].buy_orders.keys(), default=0)
        croissants_ask = min(order_depths[Product2.CROISSANTS].sell_orders.keys(), default=float("inf"))

        jams_bid = max(order_depths[Product2.JAMS].buy_orders.keys(), default=0)
        jams_ask = min(order_depths[Product2.JAMS].sell_orders.keys(), default=float("inf"))

        djembes_bid = max(order_depths[Product2.DJEMBES].buy_orders.keys(), default=0)
        djembes_ask = min(order_depths[Product2.DJEMBES].sell_orders.keys(), default=float("inf"))

        implied_bid = croissants_bid * BASKET_WEIGHTS2[Product2.CROISSANTS] + \
                       jams_bid * BASKET_WEIGHTS2[Product2.JAMS] + \
                       djembes_bid * BASKET_WEIGHTS2[Product2.DJEMBES]

        implied_ask = croissants_ask * BASKET_WEIGHTS2[Product2.CROISSANTS] + \
                       jams_ask * BASKET_WEIGHTS2[Product2.JAMS] + \
                       djembes_ask * BASKET_WEIGHTS2[Product2.DJEMBES]

        if implied_bid > 0:
            vol_bid = min(
                order_depths[Product2.CROISSANTS].buy_orders.get(croissants_bid, 0) // BASKET_WEIGHTS2[Product2.CROISSANTS],
                order_depths[Product2.JAMS].buy_orders.get(jams_bid, 0) // BASKET_WEIGHTS2[Product2.JAMS],

            )
            synthetic_depth.buy_orders[implied_bid] = vol_bid

        if implied_ask < float("inf"):
            vol_ask = min(
                -order_depths[Product2.CROISSANTS].sell_orders.get(croissants_ask, 0) // BASKET_WEIGHTS2[Product2.CROISSANTS],
                -order_depths[Product2.JAMS].sell_orders.get(jams_ask, 0) // BASKET_WEIGHTS2[Product2.JAMS],
                
            )
            synthetic_depth.sell_orders[implied_ask] = -vol_ask
        logger.print(synthetic_depth.sell_orders)
        return synthetic_depth

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = {
            Product2.CROISSANTS: [],
            Product2.JAMS: [],
            Product2.DJEMBES: [],
        }

        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity

            if quantity > 0:
                croissant_price = min(order_depths[Product2.CROISSANTS].sell_orders.keys())
                jam_price = min(order_depths[Product2.JAMS].sell_orders.keys())
                djembe_price = min(order_depths[Product2.DJEMBES].sell_orders.keys())
            else:
                croissant_price = max(order_depths[Product2.CROISSANTS].buy_orders.keys())
                jam_price = max(order_depths[Product2.JAMS].buy_orders.keys())
                djembe_price = max(order_depths[Product2.DJEMBES].buy_orders.keys())

            component_orders[Product2.CROISSANTS].append(Order(
                Product2.CROISSANTS, croissant_price, quantity * BASKET_WEIGHTS[Product2.CROISSANTS]))
            component_orders[Product2.JAMS].append(Order(
                Product2.JAMS, jam_price, quantity * BASKET_WEIGHTS[Product2.JAMS]))
            component_orders[Product2.DJEMBES].append(Order(
                Product2.DJEMBES, djembe_price, quantity * BASKET_WEIGHTS[Product2.DJEMBES]))

        return component_orders

    def execute_spread_orders(self, basket_position: int, order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]] | None:
        basket_od = order_depths[Product2.PICNIC_BASKET2]
        synthetic_od = self.get_synthetic_basket_order_depth(order_depths)

        try:
            basket_ask = min(basket_od.sell_orders)
            basket_bid = max(basket_od.buy_orders)
            synthetic_ask = min(synthetic_od.sell_orders)
            synthetic_bid = max(synthetic_od.buy_orders)
        except ValueError:
            return None  # Some order book is empty

        # --- Spread logic ---
        spread_buy = synthetic_bid - basket_ask  # We buy basket, sell synthetic
        spread_sell = basket_bid - synthetic_ask  # We sell basket, buy synthetic
     
         # Add spread to history
        if synthetic_bid and basket_ask:
            self.spread_history.append(synthetic_bid - basket_ask)

        # Default threshold fallback
        default_threshold=40

        # Dynamic threshold using std deviation
        if len(self.spread_history) >= 10:
            spread_std = np.std(self.spread_history)
            threshold = max(30, spread_std *30)  # Don't go too low
        
        else:
            threshold = default_threshold
       

        # --- Maximum allowed trade size per opportunity ---
        max_size = 10  # hard cap for safety

        # --- If synthetic overpriced, sell synthetic, buy basket ---
        if spread_buy > threshold:
            confidence = min((spread_buy - threshold) / threshold, 1.0)
            volume = int(confidence * max_size)

            # Cap volume by order book depth
            volume = min(volume, abs(basket_od.sell_orders[basket_ask]), abs(synthetic_od.buy_orders[synthetic_bid]))

            if volume <= 0:
                return None

            synthetic_orders = [Order(Product2.SYNTHETIC, synthetic_bid, -volume)]
            basket_orders = [Order(Product2.PICNIC_BASKET2, basket_ask, volume)]
            result = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            result[Product2.PICNIC_BASKET2] = basket_orders
            return result

        # --- If basket overpriced, sell basket, buy synthetic ---
        elif spread_sell > threshold:
            confidence = min((spread_sell - threshold) / threshold, 1.0)
            volume = int(confidence * max_size)

            volume = min(volume, abs(basket_od.buy_orders[basket_bid]), abs(synthetic_od.sell_orders[synthetic_ask]))

            if volume <= 0:
                return None

            synthetic_orders = [Order(Product2.SYNTHETIC, synthetic_ask, volume)]
            basket_orders = [Order(Product2.PICNIC_BASKET2, basket_bid, -volume)]
            result = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            result[Product2.PICNIC_BASKET2] = basket_orders
            return result

        return None
    def run(self, state: TradingState) -> dict[str, list[Order]]:
        basket_position = state.position.get(self.symbol, 0)
        result = self.execute_spread_orders(basket_position, state.order_depths)
        return result if result else {}
   
    def save(self) -> JSON:
        return {}

    def load(self, data: JSON) -> None:
        pass
    
# Helper functions.
def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * NormalDist().cdf(d1) - K * math.exp(-r * T) * NormalDist().cdf(d2)

def implied_volatility_bisection(S: float, K: float, T: float, market_price: float, tol: float = 1e-8, max_iter: int = 1000) -> float:
    # Define the objective: difference between theoretical and market prices.
    def objective(sigma):
        return bs_call_price(S, K, T, sigma) - market_price

    # Set the lower and upper bounds for sigma.
    lower, upper = 1e-5, 5.0
    
    # Ensure there is a sign change between the endpoints.
    if objective(lower) * objective(upper) > 0:
        # If not, return NaN to indicate that no root was bracketed.
        return float("nan")
    
    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        diff = objective(mid)
        # Check if the mid-value gives a sufficiently small error.
        if abs(diff) < tol:
            return mid
        # Narrow the interval where the sign change occurs.
        if objective(lower) * diff < 0:
            upper = mid
        else:
            lower = mid
            
    # If maximum iterations reached, return the midpoint as the best approximation.
    return (lower + upper) / 2.0

a_svi     = 0.01076
b_svi     = 0.04968
rho_svi   = 0.79812
mu_svi    = 0.19866
sigma_svi = 0.08227

def iv_svi(m: float) -> float:
    """
    Given m = log(K/S)/sqrt(T), returns implied vol from your SVI slice fit.
    """
    w = a_svi + b_svi * (
        rho_svi * (m - mu_svi)
        + math.sqrt((m - mu_svi) ** 2 + sigma_svi**2)
    )
    return math.sqrt(max(w, 0.0))


EXPIRY_DAY = 8
CURRENT_ROUND = 4

class VolatilitySpreadStrategy:
    def __init__(self, symbols: list[str], strike_prices: list[int], limits: Dict[str, int]) -> None:
        self.symbols       = symbols
        self.strike_prices = strike_prices
        self.limits         = limits
        self.orders: Dict[str, List[Order]] = {}

    def buy(self, symbol: str, price: int, quantity: int) -> None:
        # positive quantity means buy, per your conventions
        self.orders.setdefault(symbol, []).append(Order(symbol, price,  quantity))

    def sell(self, symbol: str, price: int, quantity: int) -> None:
        # positive quantity passed in, but Order expects negative for a sell
        self.orders.setdefault(symbol, []).append(Order(symbol, price, -quantity))

    def get_TTE(self, state: TradingState) -> float:
        fractional = CURRENT_ROUND + (state.timestamp / 1_000_000)
        days       = max(0.0, EXPIRY_DAY - fractional)
        return days / 365.0

    def act(self, state: TradingState) -> None:
        # 1) Fetch spot & TTE
        od0 = state.order_depths.get("VOLCANIC_ROCK")
        if not od0 or not od0.buy_orders or not od0.sell_orders:
            return
        spot_mid = (max(od0.buy_orders) + min(od0.sell_orders)) // 2
        TTE      = self.get_TTE(state)
        if TTE <= 0:
            return

        # reset orders
        self.orders = {}

        # 2) Build option‐leg trades
        for sym, K in zip(self.symbols, self.strike_prices):
            d = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue

            # fair price from SVI
            m          = math.log(K/spot_mid) / math.sqrt(TTE)
            theo_iv    = iv_svi(m)
            theo_price = BlackScholes.black_scholes_call(spot_mid, K, TTE, theo_iv)

            # determine how many we can buy/sell
            to_buy  = self.limits[sym] - state.position.get(sym, 0)
            to_sell = self.limits[sym] + state.position.get(sym, 0)

            # BUY from asks
            for ask_p, ask_v in sorted(d.sell_orders.items()):
                if to_buy <= 0 or ask_p >= theo_price:
                    break
                qty = min(-ask_v, to_buy)
                self.buy(sym, ask_p, qty)
                to_buy -= qty

            # SELL into bids
            for bid_p, bid_v in sorted(d.buy_orders.items(), reverse=True):
                if to_sell <= 0 or bid_p <= theo_price:
                    break
                qty = min(bid_v, to_sell)
                self.sell(sym, bid_p, qty)
                to_sell -= qty


        # 3) Compute net delta of the combined book
        net_delta = 0.0
        for sym, K in zip(self.symbols, self.strike_prices):
            curr    = state.position.get(sym, 0)
            traded  = sum(o.quantity for o in self.orders.get(sym, []))
            post_q  = curr + traded

            # market mid‐price for this strike
            d       = state.order_depths[sym]
            if not d.sell_orders or not d.buy_orders:
                return # skip hedging this tick
            mid     = (min(d.sell_orders) + max(d.buy_orders)) / 2

            # solve the market IV from that mid
            mkt_iv  = BlackScholes.implied_volatility(mid, spot_mid, K, TTE)
            # get the true ∂C/∂S
            dlt     = BlackScholes.delta(spot_mid, K, TTE, mkt_iv)

            net_delta += post_q * dlt

        # 4) Solve & clamp for rock hedge
        curr_rock   = state.position.get("VOLCANIC_ROCK", 0)
        change      = curr_rock - round(net_delta)
        # 5) Stepped hedge
        if change > 0:
            # BUY rock (positive change) off the ask‐side
            to_buy = min(change, 400 - curr_rock)
            for ask_p, ask_v in sorted(od0.sell_orders.items()):
                if to_buy <= 0:
                    break
                qty = min(-ask_v, to_buy)
                self.buy("VOLCANIC_ROCK", ask_p, qty)
                to_buy -= qty

        elif change < 0:
            # SELL rock (negative change) into the bid‐side
            to_sell = min(abs(change), 400 + curr_rock)
            for bid_p, bid_v in sorted(od0.buy_orders.items(), reverse=True):
                if to_sell <= 0:
                    break
                qty = min(bid_v, to_sell)
                self.sell("VOLCANIC_ROCK", bid_p, qty)
                to_sell -= qty

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        self.act(state)
        return self.orders

    def save(self) -> JSON:
        return {}

    def load(self, data: JSON) -> None:
        pass

class MacaronStrategy(Strategy):
    def __init__(self,
                 symbol: str,
                 limit: int,
                 conversion_limit: int,
                 ) -> None:
        super().__init__(symbol, limit)
        self.conversion_limit = conversion_limit

        # track the “fresh” persistence since crossing below enter_csi
        self.prev_low = False
        self.persist  = 0

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders      = []
        self.conversions = 0

        conv: ConversionObservation = state.observations.conversionObservations[self.symbol]
        self.ask       = conv.askPrice
        self.bid       = conv.bidPrice
        self.t_fees    = conv.transportFees
        self.i_tariff  = conv.importTariff
        self.e_tariff  = conv.exportTariff
        self.sun_index = conv.sunlightIndex

        depth = state.order_depths[self.symbol]
        if not depth.buy_orders or not depth.sell_orders:
            return self.orders, self.conversions
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        conv_cost = self.ask + self.t_fees + self.i_tariff
        conv_rev  = self.bid - (self.t_fees + self.e_tariff)
        pos = state.position.get(self.symbol, 0)

        # convert to other island if worth based on mid
        if pos > 0 and conv_rev > best_ask:
            qty = min(pos, self.conversion_limit)
            self.conversions -= qty
            pos -= qty
        if pos < 0 and conv_cost < best_bid:
            qty = max(-pos, -self.conversion_limit)
            self.conversions += qty
            pos += qty
        
        to_buy = self.conversion_limit - pos
        to_sell = self.conversion_limit + pos
        # sweep ask/bid if worth to convert
        for ask_p, ask_v in sorted(depth.sell_orders.items()):
            if to_buy <= 0 or ask_p >= conv_rev: 
                break
            qty = min(-ask_v, to_buy)
            self.buy(ask_p, qty)
            to_buy -= qty

        for bid_p, bid_v in sorted(depth.buy_orders.items(), reverse=True):
            if to_sell <= 0 or bid_p <= conv_cost:
                break
            qty = min(bid_v, to_sell)
            self.sell(bid_p, qty)
            to_sell -= qty
            
        if to_buy > 0:
            self.buy(best_bid + 1, to_buy)

        if to_sell > 0:
            self.sell(best_ask - 1, to_sell)
        
        return self.orders, self.conversions
    
class Trader:
    def __init__(self) -> None:
        limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "PICNIC_BASKET1": 0,
            "PICNIC_BASKET2": 0,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,
            "MAGNIFICENT_MACARONS_CONVERSIONS":10,
        }
        
        self.strategies = {
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "SQUID_INK": SquidInkStrategy("SQUID_INK", limits["SQUID_INK"]),
            "PICNIC_BASKET1": BasketTrader(Product.PICNIC_BASKET1, limits["PICNIC_BASKET1"]),
            "PICNIC_BASKET2": BasketTrader2(Product2.PICNIC_BASKET2, limits["PICNIC_BASKET2"]),
        "VOLCANIC_SPREAD": VolatilitySpreadStrategy(
                            symbols=["VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000",
                                    "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"],
                            strike_prices=[9500, 9750, 10000, 10250, 10500],
                            limits=limits,
                            ),
            "MAGNIFICIENT_MACARONS": MacaronStrategy(
                symbol="MAGNIFICENT_MACARONS",
                limit=limits["MAGNIFICENT_MACARONS"],
                conversion_limit=limits["MAGNIFICENT_MACARONS_CONVERSIONS"],
            )
        }

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
            
            if isinstance(strategy, MacaronStrategy):
                orders[strategy.symbol] = result[0]
                conversions = result[1]
            elif isinstance(result, list):  # single-symbol strategy
                orders[strategy.symbol] = result
            elif isinstance(result, dict):  # multi-symbol strategy
                orders.update(result)

            new_trader_data[key] = strategy.save()
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        #  Plot only once — at the very end of the run
       

        return orders, conversions, trader_data