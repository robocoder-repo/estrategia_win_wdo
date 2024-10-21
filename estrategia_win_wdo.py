import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class MLStrategy(bt.Strategy):
    params = (
        ('lookback', 20),
        ('rf_lookback', 100),
        ('rf_features', ['open', 'high', 'low', 'close', 'volume']),
        ('trend_lookback', 50),
        ('stop_loss_pct', 0.01),
        ('take_profit_pct', 0.02),
        ('max_holding_period', 5),
        ('volatility_threshold', 0.02),
    )

    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_volume = self.datas[0].volume

        self.sma_trend = bt.indicators.SMA(self.data_close, period=self.params.trend_lookback)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.stoch = bt.indicators.Stochastic(self.data, period=14, period_dfast=3, period_dslow=3)

        self.order = None
        self.entry_price = None
        self.entry_bar = 0

    def next(self):
        if len(self.data) < self.params.rf_lookback:
            return

        if self.order:
            return

        if self.position:
            if self.check_exit_conditions():
                self.close()

        else:
            self.train_model()
            features = self.get_features()
            prediction = self.rf_model.predict([features])[0]

            if prediction == 1 and self.check_entry_conditions():
                size = self.calculate_position_size()
                self.order = self.buy(size=size)
                self.entry_price = self.data_close[0]
                self.entry_bar = len(self)

    def train_model(self):
        df = self.get_analysis_dataframe()
        X = df[self.params.rf_features]
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.rf_model.fit(X_train, y_train)

    def get_analysis_dataframe(self):
        data = {
            'open': self.data_open.get(size=self.params.rf_lookback),
            'high': self.data_high.get(size=self.params.rf_lookback),
            'low': self.data_low.get(size=self.params.rf_lookback),
            'close': self.data_close.get(size=self.params.rf_lookback),
            'volume': self.data_volume.get(size=self.params.rf_lookback),
        }
        return pd.DataFrame(data)

    def get_features(self):
        return [self.data_open[0], self.data_high[0], self.data_low[0], self.data_close[0], self.data_volume[0]]

    def check_entry_conditions(self):
        return (
            self.data_close[0] > self.sma_trend[0] and
            self.stoch.percK[0] < 20 and
            self.stoch.percD[0] < 20 and
            self.atr[0] / self.data_close[0] < self.params.volatility_threshold
        )

    def check_exit_conditions(self):
        current_profit = (self.data_close[0] - self.entry_price) / self.entry_price
        bars_held = len(self) - self.entry_bar

        return (
            current_profit <= -self.params.stop_loss_pct or
            current_profit >= self.params.take_profit_pct or
            bars_held >= self.params.max_holding_period
        )

    def calculate_position_size(self):
        available_margin = self.broker.get_cash() * 0.5  # Use 50% of available cash as margin
        contract_value = self.data_close[0] * 0.2  # Assuming 0.2 BRL per point for WIN
        max_contracts = int(available_margin / contract_value)
        return max(1, min(max_contracts, 5))  # Limit to between 1 and 5 contracts

def run_backtest(data):
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(MLStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01% commission
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    return results

# Simulated data for demonstration purposes
def create_simulated_data():
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2023-01-01')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    prices = np.random.randn(len(dates)).cumsum() + 100
    volume = np.random.randint(1000, 10000, size=len(dates))
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(len(dates)),
        'high': prices + abs(np.random.randn(len(dates))),
        'low': prices - abs(np.random.randn(len(dates))),
        'close': prices + np.random.randn(len(dates)),
        'volume': volume
    }, index=dates)
    
    return bt.feeds.PandasData(dataname=df)

if __name__ == '__main__':
    data = create_simulated_data()
    results = run_backtest(data)
