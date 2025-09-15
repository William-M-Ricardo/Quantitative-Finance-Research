from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)


class AegisStrategy(CtaTemplate):
    """基于布林带、Aroon和ATR的量化交易策略"""

    # 策略参数
    interval: int = 15                    # K线合成的周期
    boll_window: int = 51                 # 布林带周期
    boll_dev: float = 2.5                 # 布林带标准差倍数
    aroon_window: int = 8                 # Aroon指标周期
    atr_window: int = 4                   # ATR指标周期
    risk_level: int = 200                 # 风险等级（每次交易的金额）
    trailing_long: float = 0.5            # 多头止损系数
    trailing_short: float = 0.8           # 空头止损系数

    # 策略变量
    boll_up: float = 0.0                  # 布林带上轨
    boll_down: float = 0.0                # 布林带下轨
    aroon_up: float = 0.0                 # Aroon多头指标
    aroon_down: float = 0.0               # Aroon空头指标
    atr_value: float = 0.0                # ATR值
    trading_size: int = 0                 # 每次交易的仓位

    intra_trade_high: float = 0.0         # 持仓期间的最高价
    intra_trade_low: float = 0.0          # 持仓期间的最低价
    long_stop: float = 0.0                # 多头止损价
    short_stop: float = 0.0               # 空头止损价

    parameters: list[str] = [
        "boll_window",
        "boll_dev",
        "aroon_window",
        "risk_level",
        "atr_window",
        "interval",
        "trailing_short",
        "trailing_long"
    ]

    variables: list[str] = [
        "boll_up",
        "boll_down",
        "aroon_up",
        "aroon_down",
        "atr_value",
        "intra_trade_high",
        "intra_trade_low",
        "long_stop",
        "short_stop"
    ]

    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")

        self.bg: BarGenerator = BarGenerator(self.on_bar, self.interval, self.on_window_bar)
        self.am: ArrayManager = ArrayManager()

        self.load_bar(10)

    def on_start(self) -> None:
        """策略启动回调"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """策略停止回调"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData) -> None:
        """新的Tick数据更新回调"""
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        """新的Bar数据更新回调"""
        self.bg.update_bar(bar)

    def on_window_bar(self, bar: BarData) -> None:
        """N分钟K线数据更新回调"""
        # 撤销所有未成交的订单
        self.cancel_all()

        am = self.am
        am.update_bar(bar)

        if not am.inited:
            return

        # 计算布林带、Aroon指标和ATR值
        self.calculate_indicators(am, bar)

        # 判断是否持仓，若无持仓则进行开仓操作
        if self.pos == 0:
            self.handle_no_position(bar)

        # 若持有多头仓位，设置止损点
        elif self.pos > 0:
            self.update_long_position(bar)

        # 若持有空头仓位，设置止损点
        elif self.pos < 0:
            self.update_short_position(bar)

        # 更新策略状态并同步数据
        self.put_event()
        self.sync_data()

    def calculate_indicators(self, am: ArrayManager, bar: BarData) -> None:
        """计算布林带、Aroon指标和ATR值"""
        self.boll_up, self.boll_down = am.boll(self.boll_window, self.boll_dev)
        self.aroon_up, self.aroon_down = am.aroon(self.aroon_window)
        self.atr_value = am.atr(self.atr_window)

    def handle_no_position(self, bar: BarData) -> None:
        """无持仓时根据策略判断是否开仓"""
        boll_width = self.boll_up - self.boll_down
        self.trading_size = int(self.risk_level / self.atr_value)

        if self.aroon_up > self.aroon_down:
            self.buy(self.boll_up, self.trading_size, True)
        else:
            self.short(self.boll_down, self.trading_size, True)

        self.intra_trade_high = bar.high_price
        self.intra_trade_low = bar.low_price

    def update_long_position(self, bar: BarData) -> None:
        """更新多头持仓时的止损价"""
        self.intra_trade_high = max(self.intra_trade_high, bar.high_price)
        self.intra_trade_low = bar.low_price
        boll_width = self.boll_up - self.boll_down
        self.long_stop = self.intra_trade_high - self.trailing_long * boll_width
        self.sell(self.long_stop, abs(self.pos), True)

    def update_short_position(self, bar: BarData) -> None:
        """更新空头持仓时的止损价"""
        self.intra_trade_high = bar.high_price
        self.intra_trade_low = min(self.intra_trade_low, bar.low_price)
        boll_width = self.boll_up - self.boll_down
        self.short_stop = self.intra_trade_low + self.trailing_short * boll_width
        self.cover(self.short_stop, abs(self.pos), True)

    def on_order(self, order: OrderData) -> None:
        """新的订单数据更新回调"""
        pass

    def on_trade(self, trade: TradeData) -> None:
        """新的成交数据更新回调"""
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """停止单更新回调"""
        pass
