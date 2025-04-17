
import json
from abc import abstractmethod
from math import log, sqrt, exp
import math
from statistics import NormalDist
import statistics
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List, TypeAlias
from statistics import NormalDist
from math import log, sqrt, exp
from collections import deque
from datamodel import Order, TradingState, Symbol
from collections import deque


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

from datamodel import OrderDepth, Order
from typing import List, Dict

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

    def execute_spread_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]] | None:
        logger.print("target whore")
        logger.print(target_position)
        logger.print(basket_position)
        if target_position < basket_position:
            return None

        quantity = abs(target_position - basket_position)
        basket_od = order_depths[Product.PICNIC_BASKET1]
        synthetic_od = self.get_synthetic_basket_order_depth(order_depths)
        logger.print("synthetic orders")
        logger.print(synthetic_od.sell_orders)
        logger.print(synthetic_od.buy_orders)

        try:
            basket_ask = min(basket_od.sell_orders)
            basket_bid = max(basket_od.buy_orders)
            synthetic_ask = min(synthetic_od.sell_orders)
            synthetic_bid = max(synthetic_od.buy_orders)
        except ValueError:
            return None

        if target_position > basket_position and basket_ask < synthetic_bid:
            volume = min(quantity, abs(basket_od.sell_orders[basket_ask]), abs(synthetic_od.buy_orders[synthetic_bid]))
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_ask, volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid, -volume)]
            result = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            result[Product.PICNIC_BASKET1] = basket_orders
            return result

        elif target_position < basket_position and basket_bid > synthetic_ask:
            volume = min(quantity, abs(basket_od.buy_orders[basket_bid]), abs(synthetic_od.sell_orders[synthetic_ask]))
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_bid, -volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask, volume)]
            result = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            result[Product.PICNIC_BASKET1] = basket_orders
            return result

        return None
    

# Modified Trader that integrates the pairs strategy.
class Trader:
    def __init__(self) -> None:
        self.limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }
    
        
        # Store individual strategies and, for croissants and jam, a combined pairs strategy.
        # Note: Remove the separate "CROISSANTS" and "JAMS" strategies since they are handled as a pair.
        self.spread_strategy = BasketTrader()
        logger.print("basek_pos")
        logger.print(self.spread_strategy)


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        logger.print(state.position)
        conversions = 0
        orders: Dict[str, List[Order]] = {}

        basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
        target_position = 60  # or a non-zero value if you want to take a spread position
       
        result = self.spread_strategy.execute_spread_orders(
            target_position,
            basket_position,
            state.order_depths
        )
        logger.print("results")
        logger.print(result)
        if result:
            for symbol, strat_orders in result.items():
                if symbol not in orders:
                    orders[symbol] = []
                orders[symbol].extend(strat_orders)

        trader_data = "{}"
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
