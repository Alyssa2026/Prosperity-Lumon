import json
from abc import abstractmethod
import math
from statistics import NormalDist
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from typing import Any, Dict, List, TypeAlias
from statistics import NormalDist
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

class MacaronStrategy(Strategy):
    
    STORAGE_COST_PER_LONG = 0.1
    
    def __init__(self, symbol: str, limit: int, conversion_limit: int) -> None:
        super().__init__(symbol, limit)
        self.conversion_limit = conversion_limit
    
    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0
        self.unpackObservations(state.observations.conversionObservations["MAGNIFICENT_MACARONS"])
        self.act(state)
        return self.orders, self.conversions
    
    def unpackObservations(self, conversionObservations: ConversionObservation):
        self.bidPrice = conversionObservations.bidPrice
        self.askPrice = conversionObservations.askPrice
        self.transportFees = conversionObservations.transportFees
        self.exportTariff = conversionObservations.exportTariff
        self.importTariff = conversionObservations.importTariff
        self.sugarPrice = conversionObservations.sugarPrice
        self.sunlightIndex = conversionObservations.sunlightIndex
        
    def get_predicted_future_value(self, state):
        pass
    
    def get_mid_price(self, state):
        pass
    
    def act(self, state: TradingState):
        pass
        
        
        
        
    
class Trader:
    def __init__(self) -> None:
        limits = {
            "MAGNIFICENT_MACARONS": 75,
            "MAGNIFICENT_MACARONS_CONVERSIONS":10,
        }
        
        self.strategies = {
            "MAGNIFICIENT_MACARONS": MacaronStrategy("MAGNIFICENT_MACARONS", limits["MAGNIFICENT_MACARONS"], limits["MAGNIFICENT_MACARONS_CONVERSIONS"])
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