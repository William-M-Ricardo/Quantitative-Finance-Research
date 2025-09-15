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


class AtrRsiDemo(CtaTemplate):
    """基于ATR和RSI指标的交易策略"""

    # 策略参数
    atr_window: int = 8                     # ATR指标周期
    atr_ma_window: int = 6                  # ATR均线周期
    rsi_window: int = 7                     # RSI指标周期
    rsi_entry: int = 42                     # RSI开仓阈值
    trailing_percent: float = 0.5           # 移动止损百分比
    fixed_size: int = 1                     # 固定交易数量

    # 策略变量
    atr_value: float = 0.0                  # ATR指标值
    atr_ma: float = 0.0                     # ATR均线值
    rsi_value: float = 0.0                  # RSI指标值
    rsi_long_threshold: float = 0.0         # RSI多头开仓阈值
    rsi_short_threshold: float = 0.0        # RSI空头开仓阈值
    long_trailing_target: float = 0.0       # 多头移动目标价
    short_trailing_target: float = 0.0      # 空头移动目标价
    long_trailing_distance: float = 0.0     # 多头移动距离
    short_trailing_distance: float = 0.0    # 空头移动距离
    long_trailing_stop: float = 0.0         # 多头移动止损价
    short_trailing_stop: float = 0.0        # 空头移动止损价

    # 参数名称列表
    parameters: list[str] = [
        "atr_window",
        "atr_ma_window",
        "rsi_window",
        "rsi_entry",
        "trailing_percent",
        "fixed_size"
    ]

    # 变量名称列表
    variables: list[str] = [
        "atr_value",
        "atr_ma",
        "rsi_value",
        "rsi_long_threshold",
        "rsi_short_threshold",
        "long_trailing_target",
        "short_trailing_target",
        "long_trailing_distance",
        "short_trailing_distance",
        "long_trailing_stop",
        "short_trailing_stop"
    ]

    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")

        self.bg: BarGenerator = BarGenerator(self.on_bar)
        self.am: ArrayManager = ArrayManager()

        self.rsi_long_threshold: int = 50 + self.rsi_entry
        self.rsi_short_threshold: int = 50 - self.rsi_entry

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
        # 更新K线缓存数组并检查初始化完成
        if not self.update_am(bar):
            return

        # 撤销之前的所有委托
        self.cancel_all()

        # 计算ATR和RSI指标值
        self.calculate_indicators()

        # 如果没有持仓,判断是否开仓
        if not self.pos:
            # 判断ATR趋势是否满足开仓条件
            if self.check_trend_by_atr():
                # 根据RSI指标值判断开仓方向
                self.check_entry_by_rsi(bar)

        # 更新移动止损的目标价格
        self.update_trailing_target(bar)

        # 计算移动止损的距离
        self.calculate_trailing_distance()

        # 计算移动止损的触发价格
        self.calculate_trailing_stop()

        # 发送移动止损委托
        self.send_trailing_order()

        # 推送策略状态更新事件
        self.put_event()

    def update_am(self, bar: BarData) -> bool:
        """更新K线到数据容器"""
        self.am.update_bar(bar)
        return self.am.inited

    def calculate_indicators(self) -> None:
        """计算相关技术指标"""
        atr_array = self.am.atr(self.atr_window, array=True)
        self.atr_value = atr_array[-1]
        self.atr_ma = atr_array[-self.atr_ma_window:].mean()

        self.rsi_value = self.am.rsi(self.rsi_window)

    def check_trend_by_atr(self) -> bool:
        """基于ATR指标判断趋势强度"""
        return self.atr_value > self.atr_ma

    def check_entry_by_rsi(self, bar: BarData) -> None:
        """基于RSI指标判断多空方向开仓"""
        if self.rsi_value > self.rsi_long_threshold:
            self.buy(bar.close_price + 5, self.fixed_size)
        elif self.rsi_value < self.rsi_short_threshold:
            self.short(bar.close_price - 5, self.fixed_size)

    def update_trailing_target(self, bar: BarData) -> None:
        """更新移动止损跟踪目标价"""
        if self.pos == 0:
            self.long_trailing_target = bar.high_price
            self.short_trailing_target = bar.low_price
        elif self.pos > 0:
            self.long_trailing_target = max(self.long_trailing_target, bar.high_price)
            self.short_trailing_target = bar.low_price
        elif self.pos < 0:
            self.short_trailing_target = min(self.short_trailing_target, bar.low_price)
            self.long_trailing_target = bar.high_price

    def calculate_trailing_distance(self) -> None:
        """计算移动止损距离"""
        if self.pos > 0:
            self.long_trailing_distance = self.long_trailing_target * self.trailing_percent / 100
            self.short_trailing_distance = 0
        elif self.pos < 0:
            self.long_trailing_distance = 0
            self.short_trailing_distance = self.short_trailing_target * self.trailing_percent / 100

    def calculate_trailing_stop(self) -> None:
        """计算移动止损价格"""
        if self.pos > 0:
            self.long_trailing_stop = self.long_trailing_target - self.long_trailing_distance
            self.short_trailing_stop = 0
        elif self.pos < 0:
            self.long_trailing_stop = 0
            self.short_trailing_stop = self.short_trailing_target + self.short_trailing_distance

    def send_trailing_order(self) -> None:
        """发送移动止损委托"""
        if self.pos > 0:
            self.sell(self.long_trailing_stop, abs(self.pos), stop=True)
        elif self.pos < 0:
            self.cover(self.short_trailing_stop, abs(self.pos), stop=True)

    def on_order(self, order: OrderData) -> None:
        """新的订单数据更新回调"""
        pass

    def on_trade(self, trade: TradeData) -> None:
        """新的成交数据更新回调"""
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """停止单更新回调"""
        pass
