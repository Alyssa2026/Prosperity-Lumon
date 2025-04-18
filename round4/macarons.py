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

from collections import deque
from typing import Tuple, List

class MacaronStrategy(Strategy):
    # === step‑50 intercept & coefs from your offline fit ===
    INTERCEPT_50 =  0.33327811328582885           # ← replace with your lr50.intercept_
    COEFS_50 = {
        'sugarPrice_dmean':    -10.8603,
        'exportTariff_dmean':   73.0951,
        'transportFees_dmean': 1966.8889,
        'importTariff_dmean':  -849.9047,
        'sunlightIndex_dmean': -197.4299
    }

    # how much of that predicted Δ to embed in our passives
    BIAS_FACTOR = 3

    STORAGE_COST_PER_LONG = 0.1

    def __init__(self, symbol: str, limit: int, conversion_limit: int) -> None:
        super().__init__(symbol, limit)
        self.conversion_limit = conversion_limit

        # raw features and their rolling windows (50 ticks)
        self.orig_feats = [
            'sugarPrice',
            'exportTariff',
            'transportFees',
            'importTariff',
            'sunlightIndex'
        ]
        self.feat_windows = {
            f: deque(maxlen=50) for f in self.orig_feats
        }
        self.last_feat_mean = {f: None for f in self.orig_feats}

        # a window to hold the last 50 mid_prices
        self.mid_window = deque(maxlen=50)

    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        conv = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
        self.unpackObservations(conv)

        # update feature deques
        for f in self.orig_feats:
            self.feat_windows[f].append(getattr(self, f))

        # update mid window
        mid = self.get_mid_price(state)
        if mid is not None:
            self.mid_window.append(mid)

        self.orders      = []
        self.conversions = 0
        self.act(state)
        return self.orders, self.conversions

    def unpackObservations(self, conv: ConversionObservation):
        self.bidPrice      = conv.bidPrice
        self.askPrice      = conv.askPrice
        self.transportFees = conv.transportFees
        self.exportTariff  = conv.exportTariff
        self.importTariff  = conv.importTariff
        self.sugarPrice    = conv.sugarPrice
        self.sunlightIndex = conv.sunlightIndex

    def get_mid_price(self, state: TradingState) -> float | None:
        od = state.order_depths.get(self.symbol)
        if not od or not od.buy_orders or not od.sell_orders:
            return None
        return 0.5 * (max(od.buy_orders) + min(od.sell_orders))

    def compute_dmean_features(self) -> dict:
        """
        Returns a dict of 10‑tick rolling‐mean deltas for each raw feature.
        """
        deltas = {}
        for f in self.orig_feats:
            win = self.feat_windows[f]
            if len(win) < 2:
                # not enough data yet
                deltas[f] = 0.0
            else:
                curr_mean = sum(win) / len(win)
                prev_mean = self.last_feat_mean[f]
                deltas[f] = (curr_mean - prev_mean) if prev_mean is not None else 0.0
                self.last_feat_mean[f] = curr_mean
        return deltas

    def predict_50_tick_delta(self, dmeans: dict) -> float:
        """
        Applies the hard‑coded 50‑tick regression:
          Δ̂50 = intercept + Σ coef_i * dmean_i
        """
        val = self.INTERCEPT_50
        for name, coef in self.COEFS_50.items():
            feat = name.replace('_dmean','')
            val += coef * dmeans[feat]
        return val

    def act(self, state: TradingState) -> None:
        # only proceed if we have a full 50‑tick history
        if len(self.mid_window) < 50:
            return

        # 1) get current mid
        mid = self.mid_window[-1]

        # 2) compute dmean features
        dmeans = self.compute_dmean_features()

        # 3) predict the 50‑tick change
        pred_delta = self.predict_50_tick_delta(dmeans)

        # 4) lean center of market‑making
        center = mid + self.BIAS_FACTOR * pred_delta

        # 5) post passive quotes around `center`
        od    = state.order_depths[self.symbol]
        buys  = sorted(od.buy_orders .items(), reverse=True)
        sells = sorted(od.sell_orders.items())

        pos     = state.position.get(self.symbol, 0)
        to_buy  = self.limit - pos
        to_sell = self.limit + pos

        # decide your skewed best bid/ask
        max_bid = center if pos <= 0 else center - 1
        min_ask = center if pos >= 0 else center + 1

        # passive bids
        for price, vol in sells:
            if to_buy <= 0: break
            if price <= max_bid:
                qty = min(to_buy, -vol)
                self.buy(price, qty)
                to_buy -= qty

        # passive asks
        for price, vol in buys:
            if to_sell <= 0: break
            if price >= min_ask:
                qty = min(to_sell, vol)
                self.sell(price, qty)
                to_sell -= qty

        
        
        
        
    
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