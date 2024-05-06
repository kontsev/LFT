from AlgorithmImports import *
from pykalman import KalmanFilter
import numpy as np
from datetime import datetime
import statsmodels.api as sm

class PairTradingAlgorithm(QCAlgorithm):
    def Initialize(self):
        # Настройка начальной даты, конечной даты и начального капитала
        self.SetStartDate(2020, 1, 1)  # ГГГГ, ММ, ДД
        self.SetEndDate(2021, 1, 1)    # ГГГГ, ММ, ДД
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)           # Начальный капитал

        # Добавление активов
        self.eth = self.AddCrypto("ETHUSDT", Resolution.Hour, market="Binance").Symbol
        self.qtum = self.AddCrypto("QTUMUSDT", Resolution.Hour, market="Binance").Symbol

        # Инициализация переменных для хранения настроек модели
        # Эти параметры тоже будут изначально даны, поскольку мы предобучаем модель вне класса
        self.window = None
        self.k_open = None
        self.k_close = 0
        self.open_t = None
        self.close_t = None
        self.halflife = None

        # Переменные для хранения предсказанных спредов и состояний фильтра Калмана
        self.predicted_spreads = [] # В начальный момент predicted spread будет содержать 1 элемент
        self.filtered_state_mean = None
        self.filtered_state_covariance = None

        # 0 ставится для того, чтобы для первой итерации алгоритма мы могли взять prev_spread
        self.predicted_spreads_backtest = [0]

        # Фильтр
        self.kf = None

        # Добавим исторические данные для оценки фильтра
        self.start_train = "2016-01-01 00:00:00"
        self.end_train = "2019-09-20 23:00:00"
        self.start_validation = "2019-09-21 00:00:00"
        self.end_validation = "2019-12-31 23:00:00"

        self.start_backtest = datetime.strptime("2016-01-01", "%Y-%m-%d")
        self.end_backtest = datetime.strptime("2019-12-31", "%Y-%m-%d")
        self.name_1 = "close_eth"
        self.name_2 = "close_qtum"
        self.namelog_1 = "close_eth_log"
        self.namelog_2 = "close_qtum_log"
        
        # Параметры для функции find_best_params
        self.amount = 1
        self.pairs = {(self.name_1, self.name_2)}
        self.pair = self.pairs

        # FIX: Класс QuantBook не предназначен для использования внутри QCAlgorithm
        # self.qb = QuantBook()

        # self.Log("Fetching history at init")

        self.history_eth = self.History(self.eth, self.start_backtest, self.end_backtest)
        self.history_qtum = self.History(self.qtum, self.start_backtest, self.end_backtest)

        # Здесь будут все необходимые параметры для нахождения оптимальных гиперпараметров (метод find_best_params)
        self.pairs = {('close_eth', 'close_qtum')}

    # DONE
    def train_filter(self):
        history_eth_reset = self.history_eth.reset_index()
        history_eth_filtered = history_eth_reset[['time', 'close']].copy()
        history_eth_filtered.rename(columns={'close': self.name_1}, inplace=True)

        history_qtum_reset = self.history_qtum.reset_index()
        history_qtum_filtered = history_qtum_reset[['time', 'close']].copy()
        history_qtum_filtered.rename(columns={'close': self.name_2}, inplace=True)

        combined_df = pd.merge(history_eth_filtered, history_qtum_filtered, on='time', how='inner')
        combined_df.set_index('time', inplace=True)
        combined_df[self.namelog_1] = np.log(combined_df[self.name_1])
        combined_df[self.namelog_2] = np.log(combined_df[self.name_2])

        train = combined_df.loc[self.start_train:self.end_train]
        validation = combined_df.loc[self.start_validation: self.end_validation]

        # self.Log(train.head(1))
        # self.Log(train.tail(1))
        # self.Log(validation.head(1))
        # self.Log(validation.tail(1))

        y = train[self.namelog_1]
        x = train[self.namelog_2]

        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
        obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

        self.kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
                        initial_state_mean=[0.95,0.95],
                        initial_state_covariance=np.ones((2, 2)),# 2x2 matrix of ones for covariance matrix estimate
                        transition_matrices=np.eye(2),
                        observation_matrices=obs_mat,
                        observation_covariance=2,
                        transition_covariance=trans_cov,
                        random_state=4322)

        state_means_gm, state_covs_gm = self.kf.filter(y.values)


        y_new = validation[self.namelog_1].values
        x_new = validation[self.namelog_2].values

        spreads = []

        # TODO: Понять что за дополнительные данные в y, y_new
        # Размер y и y_new в классе отличается от размера y и y_new в ноутбуке
        # В классе: len(y) = 8779, len(y_new) = 2430, 
        # В ноутбуке: len(y) = 8752, len(y_new) = 2428
        # Из-за этого нельзя сравнить результаты в ноутбуке и классе
        # self.Log(y)
        # self.Log(x)
        # self.Log(len(y_new))
        # self.Log(state_means_gm[-1, 0])
        # self.Log(state_means_gm[-1, 1])
        first_spread = y_new[0] - state_means_gm[-1, 0] * x_new[0] - state_means_gm[-1, 1]
        # self.Log(first_spread)
        self.predicted_spreads.append(first_spread)

        self.filtered_state_mean = state_means_gm[-1]
        self.filtered_state_covariance = state_covs_gm[-1]


        for t in (range(len(x_new) - 1)):
            obs_mat = np.array([[x_new[t], 1.0]]) # For current observation
            
            # Update Kalman filter
            self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
                self.filtered_state_mean, self.filtered_state_covariance, observation=y_new[t], observation_matrix=obs_mat)

            current_spread = y_new[t] - (self.filtered_state_mean[0] * x_new[t] + self.filtered_state_mean[1])
            spreads.append(current_spread)

            predicted_spread_t1 = y_new[t + 1] - self.filtered_state_mean[0] * x_new[t + 1] - self.filtered_state_mean[1]
            self.predicted_spreads.append(predicted_spread_t1)
        
        obs_mat = np.array([[x_new[len(x_new) - 1], 1.0]]) # For current observation
        self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
            self.filtered_state_mean, self.filtered_state_covariance, observation=y_new[len(y_new) - 1], observation_matrix=obs_mat)

        # TODO: Исследовать этот момент в ноутбуке, в качестве spread_prev для теста брать последний спред для валидации
        # self.predicted_spreads_backtest.append(self.predicted_spreads[-1])

        # Находим подходящие гиперпараметры по Валидации
        # TODO: Написать нахождение оптимальных гиперпараметров
        validation['spread_kalman'] = self.predicted_spreads
        self.window, self.k_open = self.find_best_params(validation)
        self.halflife = self.half_life(validation['spread_kalman'])
        # self.Log(self.window)
        # self.Log(self.k_open)
        

        


    def OnData(self, data):
        # self.Log("OnData method")

        if (self.kf is None) and (self.history_eth  is not None) and (self.history_qtum is not None):
            # self.Log("Train Kalman filter")
            self.train_filter()
            # self.Log("Done training")

        # # Пример получения исторических данных в OnData
        # current_time = self.Time
        # history_start_time = self.Time - timedelta(days=2) 

        # history_eth = self.History(self.eth, history_start_time, current_time)
        # history_qtum = self.History(self.qtum, history_start_time, current_time)

        # self.Log(str(history_eth))
        # self.Log(str(history_qtum))

        # # Ещё вариант: использование текущей даты и времени для запроса исторических данных
        # history_eth = self.History(["ETHUSDT"], 10, Resolution.Daily)
        # self.Log(str(history_eth))


        # Проверка наличия необходимых данных
        if not (data.ContainsKey(self.eth) and data.ContainsKey(self.qtum)):
            return

        # self.Debug(data[self.eth])
        if data[self.eth].Close == 0 or data[self.qtum].Close == 0:
            return  # Если актуальных цен нет, выходим из функции
        # Каждый раз когда приходят новые данные необходимо обновить фильтр для следующего предсказания
        self.UpdateModel(data)

    #     # Логика торговли
        self.Trade()

    # Этот метод вызывается каждый раз при получении данных с биржи
    def UpdateModel(self, data):
        # Получаем цены закрытия для каждого актива
        price_eth = np.log(data[self.eth].Close)
        price_qtum = np.log(data[self.qtum].Close)

        if len(self.predicted_spreads_backtest) < 2:
            first_spread = price_eth - self.filtered_state_mean[0] * price_qtum - self.filtered_state_mean[1]
            self.predicted_spreads_backtest.append(first_spread)
        else:
            # Обновление Калмановского фильтра
            # self.Log("In else statement Update Model")
            self.kalman_filter_update(price_eth, price_qtum)
        
            # # Оптимизация параметров стратегии
            # # Предполагается, что функция optimize_strategy_parameters уже реализована
            # # self.window, self.k_open = self.optimize_strategy_parameters(data)

        # Расчет open_t и close_t на основе обновленных параметров
        rolling_spread = np.array(pd.Series(self.predicted_spreads_backtest).rolling(window=self.window).mean())
        # rolling_spread = np.mean(self.predicted_spreads_backtest[:len(self.predicted_spreads)])  # Среднее значение спреда за последние 'window' периодов
        self.open_t = rolling_spread[-1] + self.k_open
        self.close_t = rolling_spread[-1] + self.k_close
        # self.Log(self.open_t)
        # self.Log(self.close_t)

    def Trade(self):
        # Получаем текущий спред и предыдущий спред
        # current_spread = self.predicted_spreads_backtest[-1]  # Текущий предсказанный спред
        # previous_spread = self.predicted_spreads_backtest[-2] if len(self.predicted_spreads_backtest) > 1 else current_spread

        meanSpread = np.array(pd.Series(self.predicted_spreads_backtest).rolling(window=self.halflife).mean())
        stdSpread = np.array(pd.Series(self.predicted_spreads_backtest).rolling(window=self.halflife).std())
        spread_values = (self.predicted_spreads_backtest - meanSpread)/stdSpread

        current_spread = spread_values[-1]
        previous_spread = spread_values[-2]

        # self.Log(current_spread)
        # self.Log(previous_spread)

        # Получаем текущее значение open_t и close_t (здесь используются упрощенные примеры)
        open_t = self.open_t  # Предполагается, что open_t уже рассчитан
        close_t = self.close_t  # Предполагается, что close_t уже рассчитан

        # Получаем текущую позицию по ETH
        pos = self.Portfolio[self.eth].Quantity

        # Правила открытия сделки
        if current_spread >= open_t and pos == 0 and previous_spread < open_t:
            # Открытие длинной позиции по ETH
            # Здесь нужно добавить стоплосс
            self.SetHoldings(self.eth, -0.5)
            self.SetHoldings(self.qtum, 0.5)
            # self.Debug(f"Открытие длинной позиции по ETH на {self.Time}")
        elif current_spread <= -open_t and pos == 0 and previous_spread > -open_t:
            # Открытие короткой позиции по ETH
            # Здесь нужно добавить стоплосс
            self.SetHoldings(self.eth, 0.5)
            self.SetHoldings(self.qtum, -0.5)
            # self.Debug(f"Открытие короткой позиции по ETH на {self.Time}")

        # Правила закрытия сделки
        elif (current_spread <= close_t and pos > 0) or (current_spread >= -close_t and pos < 0):
            # Закрытие позиции по ETH
            self.Liquidate(self.eth)
            self.Liquidate(self.qtum)
            self.Debug(f"Закрытие позиции: {self.Time}")
    
    def kalman_filter_update(self, price_eth, price_qtum):
        # Реализуйте логику обновления Калмановского фильтра здесь
        # Возвращайте обновленное среднее состояние и ковариацию
        # self.Log("In kalman_filter_update")
        obs_mat = np.array([[price_qtum, 1.0]]) # For current observation
        
        # Update Kalman filter
        self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
            self.filtered_state_mean, self.filtered_state_covariance, observation=price_eth, observation_matrix=obs_mat)

        predicted_spread_t1 = price_eth - self.filtered_state_mean[0] * price_qtum - self.filtered_state_mean[1]
        self.predicted_spreads_backtest.append(predicted_spread_t1)
        # return updated_state_mean, updated_state_covariance

    # # def optimize_strategy_parameters(self, data):
    # #     # Реализуйте логику оптимизации параметров стратегии здесь
    # #     # Возвращайте оптимизированные значения window и k_open
    # #     return optimized_window, optimized_k_open
    def half_life(self, spread):
    
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        spread_ret = spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]
        spread_lag2 = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret,spread_lag2)
        res = model.fit()
        halflife = int(round(-np.log(2) / res.params[1],0))
        
        if halflife <= 0:
            halflife = 1
            
        return halflife
    
    def pair_strategy_modified(self, curr_trade: list, pos: float, index: int, spread: pd.Series, data: pd.DataFrame, t_cols: list, spred_prev: pd.Series) -> tuple:
        testov = spred_prev
        if spread >= data[t_cols[0]][index] and pos == 0 and testov < data[t_cols[0]][index]:  # open long
            pos = self.amount
            desicion = 'open'
        elif spread <= -data[t_cols[0]][index] and pos == 0 and testov > -data[t_cols[0]][index]:  # open short
            pos = -1 * self.amount
            desicion = 'open'
        elif (spread <= data[t_cols[1]][index] and pos > 0) or (spread >= -data[t_cols[1]][index] and pos < 0):  # close
            pos = 0
            desicion = 'close'
        else:
            desicion = 'hold'
        return desicion, pos
    def run_pair_strategy(self, df: pd.DataFrame, price_cols: list, spread_col: str, amount: float, w, strategy: 'function', t_cols: list = None, type: str = None, verbose: bool = True, viz: bool = True, verbose_print: bool = True) -> tuple:
        if t_cols is None:
            t_cols = ['open_t', 'close_t']
        # spread_values = df[spread_col].to_numpy()
        amount = self.amount

        halflife = self.half_life(df[spread_col].fillna(0))
        # # calculate z-score with window = half life period
        meanSpread = df[spread_col].rolling(window=halflife).mean()
        stdSpread = df[spread_col].rolling(window=halflife).std()
        spread_values = (df[spread_col]-meanSpread)/stdSpread
        spread_values = spread_values.to_numpy()
        
        amounts_vector = np.array([amount, -w * amount])
        prices = df[price_cols].to_numpy()
        prices_to_order = prices * amounts_vector  # цены для покупки / продажи, учитывая сторону сделки
        
        pnls = []
        open_pnls = []
        trades = []
        all_pos = []
        last_trade = None
        pos = 0
        prev_spread = 0

        for index, spread in enumerate(spread_values):
            curr_trade = prices_to_order[index]
            curr_opnl = 0
            curr_pnl = 0
            
            desicion, new_pos = strategy(curr_trade, pos, index, spread, df, t_cols, prev_spread)
            prev_spread = spread

            # считаем либо close pnl, либо open pnl
            if desicion == 'open':
                last_trade = curr_trade
                pos = new_pos
            elif desicion == 'close':
                curr_pnl = self.calc_pnl(last_trade, curr_trade, pos)
                last_trade = None
                pos = 0
            else:
                curr_opnl = self.calc_pnl(last_trade, curr_trade, pos)
            
            open_pnls.append(curr_opnl)
            pnls.append(curr_pnl)
            trades.append(desicion)
            all_pos.append(pos)
            

        pnls = np.array(pnls)
        open_pnls = np.array(open_pnls)
        trades = np.array(trades)
        
        total_df = df[price_cols].copy()
        total_df['pnl'] = pnls
        total_df['open_pnl'] = open_pnls
        total_df['trades'] = trades
        total_df['position'] = all_pos
        total_df['spread'] = df[spread_col]
        total_df['real_pnl'] = np.cumsum(pnls) + open_pnls
        total_df[t_cols] = df[t_cols]
        
        
        if verbose:
            final_pnl = total_df['real_pnl'].iloc[-1]
            pnl_per_day = final_pnl / (total_df.index[-1] - total_df.index[0]).days
            trades = (total_df['trades'] != 'hold').sum() // 2  # считаем одним трейдом пару открытие-закрытие позиции
            sharpe_ratio = self.calc_sharpe(total_df['real_pnl'])
            max_dd = self.calc_max_drawdown(total_df['real_pnl'])
            pnl_to_max_dd = final_pnl / max_dd
            summary = {f'final_pnl {self.pair}': final_pnl, 'pnl_per_day': pnl_per_day, 'trades': trades, 'max_dd': max_dd, 
                    'sharpe_ratio': sharpe_ratio, 'pnl_to_max_dd': pnl_to_max_dd}
            if verbose_print:
                print(*[f'{key}: {round(value, 4)}' for key, value in summary.items()], sep='\n')
        
        if viz:
            plot_results(total_df, price_cols, t_cols)
            
        return total_df, summary
    
    def calc_pnl(self, last_amount, curr_amount, pos):
        if pos > 0:
            pnl = np.sum(curr_amount - last_amount)
        elif pos < 0:
            pnl = np.sum(last_amount - curr_amount)
        else:
            pnl = 0
        return pnl
    
    def calc_sharpe(self, pnl: pd.Series) -> float:
        # посчитаем шарп как дельту пнлей, а не через доходности, так как у нас нет формируемого портфеля активов
        
        returns = pnl.diff().fillna(value=0)
        mean_return = returns.mean()
        std = returns.std()
        sharpe = (mean_return) / std * np.sqrt(365 * 25)
        return sharpe
    
    def calc_max_drawdown(self, pnl: pd.Series) -> float:
        max_pnl = 0
        index_max_pnl = 0
        max_drawdown = 0
        start_index = 0
        end_index = 0
        for index, value in enumerate(pnl.values):
            if value > max_pnl:
                max_pnl = value
                index_max_pnl = index
            else:
                curr_drawdown = max_pnl - value
                if curr_drawdown > max_drawdown:
                    max_drawdown = curr_drawdown
                    start_index = index_max_pnl
                    end_index = index

        return max_drawdown

    def find_best_params(self, data):
        sharpe = 0
        pnl_to_drawdown = 0
        k_close = 0
        best_window = 0
        best_k_open = 0
        list_kopen = np.arange(0.001, 0.101, 0.001)
        for window_ in range(1, 15):
            for k_open_ in list_kopen:
                amount = 1
                weight = 1

                window = 24 * window_
                for pair in self.pairs:
                    rolling_spread = data[f'spread_kalman'].rolling(window=window)

                    open_t = rolling_spread.mean() + k_open_
                    close_t = rolling_spread.mean() + k_close

                    data[f'open_t'] = open_t
                    data[f'close_t'] = close_t

                for pair in self.pairs:
                    _, summary = self.run_pair_strategy(data, [pair[1], pair[0]], f'spread_kalman', amount, weight, self.pair_strategy_modified, 
                                                [f'open_t', f'close_t'], verbose=True, viz=False, verbose_print=False)
                
                sharpe_ratio = summary['sharpe_ratio']
                # print(sharpe_ratio)
                drawdown = summary['pnl_to_max_dd']
                # print(drawdown)
                if drawdown > pnl_to_drawdown and sharpe_ratio > sharpe:
                    sharpe = sharpe_ratio
                    pnl_to_drawdown = drawdown
                    best_window = window_
                    best_k_open = k_open_
                    # print(sharpe, pnl_to_drawdown, best_k_open, best_window)
        
        
        return 24 * best_window, best_k_open