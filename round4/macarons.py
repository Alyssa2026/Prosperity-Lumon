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

from datamodel import Order, TradingState, ConversionObservation, Symbol


import math
from collections import deque
from datamodel import ConversionObservation, Order, TradingState, Symbol

class MacaronStrategy(Strategy):
    STORAGE_COST_PER_LONG = 0.1

    # scaler & logistic params from notebook
    SC_MEANS  = {'sunlightIndex': 55.1674,
                 'sun_slope':     -0.000167,
                 'low60':          0.606233,
                 'persist':       241.0499}
    SC_SCALES = {'sunlightIndex': 10.3269,
                 'sun_slope':      0.01118,
                 'low60':          0.488584,
                 'persist':       556.6172}
    COEFS     = {'sunlightIndex': -8.52,
                 'sun_slope':     -1.27,
                 'low60':         -0.04,
                 'persist':        0.05}
    INTERCEPT = -0.33

    def __init__(self,
                 symbol: Symbol,
                 limit: int,
                 conversion_limit: int,
                 post_regime_duration: int = 17):
        super().__init__(symbol, limit)
        self.conversion_limit = conversion_limit
        # tracking
        self.prev_low       = False
        self.persist_count  = 0
        self.prev_sun_index = None
        self.prev_in_regime = False
        self.cooldown       = 0
        self.post_duration  = post_regime_duration

    def run(self, state: TradingState):
        self.orders      = []
        self.conversions = 0

        # unpack conversion obs
        conv: ConversionObservation = state.observations.conversionObservations[self.symbol]
        self.ask       = conv.askPrice
        self.bid       = conv.bidPrice
        self.t_fees    = conv.transportFees
        self.i_tariff  = conv.importTariff
        self.e_tariff  = conv.exportTariff
        self.sun_index = conv.sunlightIndex

        # compute sun slope
        if self.prev_sun_index is None:
            sun_slope = 0.0
        else:
            sun_slope = self.sun_index - self.prev_sun_index
        self.prev_sun_index = self.sun_index

        # update persistence for logistic features
        ENTER_CSI, EXIT_CSI = 45, 47
        low_flag = (self.sun_index <= ENTER_CSI)
        if low_flag and not self.prev_low:
            self.persist_count = 1
        elif low_flag and self.prev_low:
            self.persist_count += 1
        else:
            if self.sun_index > EXIT_CSI:
                self.persist_count = 0
        self.prev_low = low_flag

        # scale features
        f_si  = (self.sun_index   - self.SC_MEANS['sunlightIndex']) / self.SC_SCALES['sunlightIndex']
        f_ss  = (sun_slope        - self.SC_MEANS['sun_slope'])     / self.SC_SCALES['sun_slope']
        low60 = 1 if self.sun_index < 60 else 0
        f_l60 = (low60           - self.SC_MEANS['low60'])        / self.SC_SCALES['low60']
        f_pr  = (self.persist_count - self.SC_MEANS['persist'])  / self.SC_SCALES['persist']

        # logistic regime probability
        logit = (self.INTERCEPT
                 + self.COEFS['sunlightIndex'] * f_si
                 + self.COEFS['sun_slope']     * f_ss
                 + self.COEFS['low60']         * f_l60
                 + self.COEFS['persist']       * f_pr)
        prob      = 1 / (1 + math.exp(-logit))
        in_regime = prob > 0.5

        # detect regime end and trigger cooldown
        if self.prev_in_regime and not in_regime:
            self.cooldown = self.post_duration
        self.prev_in_regime = in_regime

        depth = state.order_depths[self.symbol]
        if not depth.buy_orders or not depth.sell_orders:
            return self.orders, self.conversions

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        pos      = state.position.get(self.symbol, 0)

        # === regime actions ===
        if in_regime:
            # buy all asks to go full long
            to_buy = self.limit - pos
            for p, v in sorted(depth.sell_orders.items()):
                if to_buy <= 0:
                    break
                qty = min(-v, to_buy)
                self.buy(p, qty)
                to_buy -= qty
            return self.orders, self.conversions

        # if in cooldown, short all the way
        if self.cooldown > 0:
            to_sell = self.limit + pos
            for p, v in sorted(depth.buy_orders.items(), reverse=True):
                if to_sell <= 0:
                    break
                qty = min(v, to_sell)
                self.sell(p, qty)
                to_sell -= qty
            self.cooldown -= 1
            return self.orders, self.conversions

        # === normal market making & chef conversion ===
        # flatten via chef if profitable
        conv_cost = self.ask + self.t_fees + self.i_tariff
        conv_rev  = self.bid - self.t_fees - self.e_tariff
        mid       = 0.5 * (best_bid + best_ask)

        # flatten long
        if pos > 0 and conv_rev > mid:
            qty = min(pos, self.conversion_limit)
            self.conversions -= qty
            pos -= qty
        # flatten short
        if pos < 0 and conv_cost < mid:
            qty = min(-pos, self.conversion_limit)
            self.conversions += qty
            pos += qty

        # inside‐spread market making
        bid_cap = self.limit - pos - self.conversions
        ask_cap = self.limit + pos + self.conversions
        if bid_cap > 0:
            self.buy(best_bid + 1, bid_cap)
        if ask_cap > 0:
            self.sell(best_ask - 1, ask_cap)

        return self.orders, self.conversions


        
        
        
        
    
class Trader:
    def __init__(self) -> None:
        limits = {
            "MAGNIFICENT_MACARONS": 75,
            "MAGNIFICENT_MACARONS_CONVERSIONS":10,
        }
        
        self.strategies = {
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
       

        return orders, conversions, trader_data