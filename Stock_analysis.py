import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def menu():
    choice = 1
    while choice != 0:
        print("1. Input data\n"
              "2. Analysis\n"
              "3. Technical indicators\n"
              "0. Close program\n"
              "Please input number: ")
        choice = input()
        try:
            if choice == '1':
                print('1. Enter the required data for analysis:')
                data, stock = input_data()
            elif choice == '2':
                print('2. Analysis')
                analysis(data)
            elif choice == '3':
                print('3. Technical indicators')
                technical_indicators(data, stock)
            elif choice == '0':
                print('Goodbye!')
                return False
            else:
                print("Error!")
        except UnboundLocalError:
            print("Please enter the initial data.")
            min_periods, data, stock = input_data()
        # except ValueError:
        #     print("Please input the number, not text.")

def input_data():
    stock = input("The company stock ticker: ")
    stock = 'SBER.ME'
    start_date = input("Start date of period(*You can leave it empty. Will be set last year): ")
    start_date = '2020-06-20'
    end_date = input("End date of period(*You can leave it empty. Will be set today's date): ")
    if end_date == '':
        end_date = date.today()
    elif start_date == '':
        start_date = date(end_date.year - 1, end_date.month, end_date.day)
    try:
        data = yf.download(stock, start_date, end_date)
        print("The data was succesfully loaded.\n")
    except Exception as error:
        print("Type error: {}".format(str(error)))
    return (data, stock)

def technical_indicators(data, stock):
    choice = 1
    while choice != 0:
        print("Select the technical indicator that you want to calculate:\n"
              "1) Moving Averages\n"
              "2) RSI(Relative Strength Index)\n"
              "3) Stochastic\n"
              "4) MACD(Moving Average Convergence Divergence)\n"
              "5) Volume\n"
              "6) Rate of Change(ROC)\n"
              "7) On Balance Volume(OBV)\n"
              "0) Back to the main menu\n")
        choice = int(input())
        while not (choice < 10 and choice >= 0):
            print("Please enter the correct value. This value {} is not correct.")
            choice = int(input())
        try:
            if choice == 1:
                print('1. Moving Averages')
                moving_averages(data)
            elif choice == 2:
                print('2. RSI(Relative Strength Index)')
                rsi(data[['Adj Close']], 14)
            elif choice == 3:
                print('3. Stochastic')
                stochastic(data)
            elif choice == 4:
                print('4. MACD')
                macd(data, stock)
            elif choice == 5:
                print('5. Volume')
                volume(data, stock)
                pass
            elif choice == 6:
                print('6. Rate of Change(ROC)')
                roc(data, 5)
            elif choice == 7:
                print('7. On Balance Volume(OBV)')
                obv(data, stock)
                buy_sell(data, stock)
            elif choice == 0:
                return False
        except UnboundLocalError:
            print("Please enter the initial data.")
        except ValueError:
            print("Please input the number, not text.")
        except Exception as err:
            print("Error {}! Please try again.".format(err))

def analysis_yield(data):
    adj_сlosing_price = data[['Adj Close']]  # Скорректированая цена закрытия
    daily_yield = adj_сlosing_price.pct_change()  # Дневная доходность в процентном соотношении
    daily_yield.fillna(0, inplace=True)  # Замена значений NA на 0
    data['daily_yield'] = daily_yield # Добавление значений в таблицу
    daily_log_yield = np.log(adj_сlosing_price.pct_change() + 1)  # Дневная логарифмическая доходность
    monthly = data.resample('BM').apply(lambda x: x[-1])  # Значения за последний рабочий день месяца
    quarter = data.resample("Q").mean()  # Расчёт по кварталам и взять среднее значение за квартал
    cum_daily_return = (1 + daily_log_yield).cumprod()  # Дневная кумулятивная доходность
    cum_monthly_return = cum_daily_return.resample("M").mean()  # Средняя месячная кумулятивная доходность

    print(daily_yield.head(5))
    print(daily_log_yield.head(5))
    print(monthly.pct_change().tail())  # Месячная доходность
    print(quarter.pct_change().tail())  # Квартальная доходность
    print(cum_daily_return.tail())  # Кумулятивная дневная доходность
    print(cum_monthly_return.tail())  # Кумулятивная месячная доходность

    cum_daily_return.plot(figsize=(12, 6))
    plt.show()
    daily_log_yield.hist(bins=50)
    plt.show()

    return False

def moving_averages(data):
    choice = '1'
    i = 0
    type_prices = ['Low', 'Medium', 'High', 'Close', 'Adj Close']
    quantity_rows = 10
    while choice != '0':
        print("Select the type of moving average:\n"
            "1) SMA – Simple Moving Average\n"
            "2) WMA – Weighted Moving Average\n"
            "3) EMA – Exponential Moving Average\n"
            "4) All Moving Averages together\n"
            "5) Exit to the main menu")
        choice = input('Enter which item to run:\n')
        if choice == '5':
            return False
        n = int(input('Enter the period for which the moving average will be calculated:\n'))
        while not (i > 0 or i <= 5):
            try:
                i = int(input('1. Open\n2. High\n3. Low\n4. Close\n5. Adj Close:\nPlease select the type price: '))
                type_price = type_prices[i - 1]
            except (IndexError, ValueError):
                print("The value {} is not correct. Please re-enter.".format(i))
        type_price = type_prices[4]
        if choice == '1':
            sma(data, type_price, n)
        if choice == '2':
            wma(data, type_price, n)
        if choice == '3':
            ema(data, type_price, n)
        if choice == '4':
            sma(data, type_price, n)
            wma(data, type_price, n)
            ema(data, type_price, n)
            plt.figure(figsize=(12, 6))
            plt.plot(data[type_price], label="Price")
            plt.plot(sma, label="SMA")
            #plt.plot(sma, label="SMA")
            plt.plot(ema, label="EMA")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.show()

    data[['Adj Close', 'moving_avg40']].plot(figsize=(5, 5))  # Построение полученных значений
    plt.show()
    return False

def sma(data, type_price, n):
    sma = data[type_price].rolling(window=n).mean()  # Простая скользящая средняя
    data['SMA'] = np.round(sma, decimals=3)
    print(data[[type_price, 'SMA']].head(10))
    print(data[[type_price, 'SMA']].tail(10))
    data[['Adj Close', 'SMA']].plot(figsize=(12, 6))
    plt.title("SMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()

def ema(data, type_price, n):
    ema = data[type_price].ewm(span=n).mean()
    data['EMA'] = np.round(ema, decimals=3)
    print(data[[type_price, 'EMA']].head(10))
    print(data[[type_price, 'EMA']].tail(10))
    data[['Adj Close', 'EMA']].plot(figsize=(12, 6))
    plt.title("EMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()

def wma(data, type_price, n):
    weights = np.arange(1, n+1, 1)
    wma = data[type_price].rolling(window=n).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    data['WMA'] = np.round(wma, decimals=3)
    data[[type_price, 'WMA']].head(10)
    data[[type_price, 'WMA']].tail(10)
    data[['Adj Close', 'WMA']].plot(figsize=(12, 6))
    plt.title("WMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()

def rsi(data, time_window):
    diff = data.diff(1).dropna()  # Вычисление разницы между n и n-1 значениями и удаление NA значений

    # Замена всех значений на 0 с сохранием знака + или -
    up_chg = 0 * diff
    down_chg = 0 * diff

    # Замена плюсовых значений на положительные значения из массива diff, в ином случае остаётся 0
    up_chg[diff > 0] = diff[diff > 0]

    # Замена минусовых значений на отрицательные значения из массива diff, остальные значения уже заменены
    down_chg[diff < 0] = diff[diff < 0]

    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

    # Расчёт значений rsi
    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    data['rsi'] = rsi

    # Вывод начала и коцна таблицы
    print(data.head())
    print(data.tail())

    # Построение графика
    plt.figure(figsize=(15, 5))
    plt.plot(data['Adj Close'])
    plt.title('Price chart (Adj Close)')
    plt.show()

    # plot correspondingRSI values and significant levels
    plt.figure(figsize=(15, 5))
    plt.title('rsi chart')
    plt.plot(data['rsi'])

    plt.axhline(0, linestyle='--', alpha=0.1)
    plt.axhline(20, linestyle='--', alpha=0.5)
    plt.axhline(30, linestyle='--')

    plt.axhline(70, linestyle='--')
    plt.axhline(80, linestyle='--', alpha=0.5)
    plt.axhline(100, linestyle='--', alpha=0.1)
    plt.show()

def macd(data, stock):
    df = data[['Close']].copy()
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Close']

    # Calculate the short and long term exponential moving average (EMA)
    ShortEMA = df.Close.ewm(span=12, adjust=False).mean()
    LongEMA = df.Close.ewm(span=26, adjust=False).mean()
    # Calculate the MACD line
    macd = ShortEMA - LongEMA
    # Calculate the signal line
    signal = macd.ewm(span=9, adjust=False).mean()

    df['MACD'] = macd
    df['Signal line'] = signal

    print(data.head())
    print(data.tail())

    plt.figure(figsize=(15, 6))
    plt.plot(df.Date, macd, label='{} MACD'.format(stock), color='#EBD2BE')
    plt.plot(df.Date, signal, label='Signal Line', color='#E5A4CB')
    plt.legend(loc='upper left')
    plt.show()

    def buy_sell(df):
        buy = []
        sell = []
        flag = -1

        for i in range(0, len(df)):
            if df['MACD'][i] > df['Signal line'][i]:
                sell.append(np.nan)
                if flag != 1:
                    buy.append(df['Close'][i])
                    flag = 1
                else:
                    buy.append(np.nan)
            elif df['MACD'][i] < df['Signal line'][i]:
                buy.append(np.nan)
                if flag != 0:
                    sell.append(df['Close'][i])
                    flag = 0
                else:
                    sell.append(np.nan)
            else:
                buy.append(np.nan)
                sell.append(np.nan)

        df['Buy_Signal_Price'] = buy
        df['Sell_Signal_Price'] = sell

        plt.figure(figsize=(15, 6))
        plt.scatter(df.index, df['Buy_Signal_Price'], color='green', label='Buy', marker='^', alpha=1)
        plt.scatter(df.index, df['Sell_Signal_Price'], color='red', label='Sell', marker='v', alpha=1)
        plt.plot(df['Close'], label='Close price', alpha=0.35)
        plt.title('Close Price Buy & Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend(loc='upper left')
        plt.show()

    buy_sell(df)

def stochastic(data):
    data['14-high'] = data['High'].rolling(14).max()
    data['14-low'] = data['Low'].rolling(14).min()
    data['%K'] = (data['Close'] - data['14-low']) * 100 / (data['14-high'] - data['14-low'])
    data['%D'] = data['%K'].rolling(3).mean()

    data[['High', 'Low', 'Close', '%K', '%D']].head(10)
    data[['High', 'Low', 'Close', '%K', '%D']].tail(10)

    ax = data[['%K', '%D']].plot()
    data['Adj Close'].plot(ax=ax, secondary_y=True)
    ax.axhline(20, linestyle='--', color="r")
    ax.axhline(80, linestyle="--", color="r")
    plt.show()

def volume(data, stock):
    n = 5
    data['Momentum'] = data['Adj Close']/data['Adj Close'].shift(n) * 100

def momentum(data, stock):
    n = 5
    data['Momentum'] = data['Adj Close'] / data['Adj Close'].shift(n) * 100

# On Balance Volume
def obv(data, stock):
    OBV = []
    OBV.append(0)
    for i in range(1, len(data.Close)):
        if data.Close[i] > data.Close[i - 1]:  # If the closing price is above the prior close price
            OBV.append(OBV[-1] + data.Volume[i])  # then: Current OBV = Previous OBV + Current Volume
        elif data.Close[i] < data.Close[i - 1]:
            OBV.append(OBV[-1] - data.Volume[i])
        else:
            OBV.append(OBV[-1])

    # Store the OBV and OBV EMA into new columns
    data['OBV'] = OBV
    data['OBV_EMA'] = data['OBV'].ewm(com=20).mean()
    # Show the data
    print(data[['Adj Close', 'OBV', 'OBV_EMA']].head())
    print(data[['Adj Close', 'OBV', 'OBV_EMA']].tail())

    # Create and plot the graph
    plt.figure(figsize=(15, 6))
    plt.plot(data['OBV'], label='OBV', color='orange')
    plt.plot(data['OBV_EMA'], label='OBV_EMA', color='purple')
    plt.xticks(rotation=45)
    plt.title('OBV/OBV_EMA {}'.format(stock))
    plt.xlabel('Дата', fontsize=18)
    plt.ylabel('Цена', fontsize=18)
    plt.show()

def buy_sell(data, stock):
    signal = data
    sigPriceBuy = []
    sigPriceSell = []
    flag = -1  # A flag for the trend upward/downward
    # Loop through the length of the data set
    for i in range(0, len(signal)):
        # if OBV > OBV_EMA  and flag != 1 then buy else sell
        if data.OBV[i] > data.OBV_EMA[i] and flag != 1:
            sigPriceBuy.append(signal['Close'][i])
            sigPriceSell.append(np.nan)
            flag = 1
        # else  if OBV < OBV_EMA  and flag != 0 then sell else buy
        elif data.OBV[i] < data.OBV_EMA[i] and flag != 0:
            sigPriceSell.append(signal['Close'][i])
            sigPriceBuy.append(np.nan)
            flag = 0
        # else   OBV == OBV_EMA  so append NaN
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    data['Buy_Signal_Price'] = sigPriceBuy
    data['Sell_Signal_Price'] = sigPriceSell

    plt.figure(figsize=(15, 6))
    plt.scatter(data.index, data['Buy_Signal_Price'], color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(data.index, data['Sell_Signal_Price'], color='red', label='Sell Signal', marker='v', alpha=1)
    plt.plot(data['Close'], label='Close Price', alpha=0.35)
    plt.xticks(rotation=45)
    plt.title('Сигналы на покупку и продажу акции {}'.format(stock))
    plt.xlabel('Дата', fontsize=18)
    plt.ylabel('Цена', fontsize=18)
    plt.legend(loc='upper left')
    plt.show()

# Rate of Change (ROC)
def roc(data, n):
    n = 5
    momentum = data['Adj Close'].diff(n)
    p = data['Adj Close'].shift(n)
    compute = pd.Series((momentum / p) * 100, name='ROC {}'.format(n))
    data = data.join(compute)

    print(data.head(n + 5))
    print(data.tail())

    # Plotting the Price Series chart and the Ease Of Movement below
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xticklabels([])
    plt.plot(data['Adj Close'], lw=1)
    plt.title('NSE Price Chart')
    plt.ylabel('Close Price')
    plt.grid(True)
    bx = fig.add_subplot(2, 1, 2)
    plt.plot(compute, 'k', lw=0.75, linestyle='-', label='ROC')
    plt.legend(loc=2, prop={'size': 9})
    plt.ylabel('ROC values')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)


def analysis(data):
    min_periods = int(input())
    daily_yield = data[['Adj Close']].pct_change()  # Дневная доходность в процентом соотношении
    daily_yield.fillna(0, inplace=True)
    daily_log_yield = np.log(daily_yield.pct_change() + 1)  # Логарифмическая дневная доходность
    daily_log_yield.fillna(0, inplace=True)

    vol = daily_yield.rolling(min_periods).std() * np.sqrt(min_periods)  # Волатильность

    print(daily_yield.head())
    print(daily_yield.tail())
    print(daily_log_yield.head())
    print(daily_log_yield.tail())

    monthly = data[['Adj Close']].resample('BM').apply(lambda x: x[-1])  # Взять значения за последний рабочий день месяца
    quarter = data[['Adj Close']].resample("4M").mean()  # Пересчитать `sber` по кварталам и взять среднее значение за квартал
    cum_daily_return = (1 + daily_yield).cumprod()  # Кумулятинвая доходность
    cum_monthly_return = cum_daily_return.resample("M").mean()

    print(monthly.pct_change().head())
    print(monthly.pct_change().tail())  # Месячная доходность
    print(quarter.pct_change().head())
    print(quarter.pct_change().tail())  # Квартальную доходность
    print(cum_daily_return.head())
    print(cum_daily_return.tail())  # Кумулятивная дневная доходность
    print(cum_monthly_return.head())
    print(cum_monthly_return.tail())  # Кумулятивная месячная доходность

    daily_yield.hist(bins=50)
    plt.show()

    cum_daily_return.plot(figsize=(12, 8))
    plt.show()

    # Общая статистика
    print(daily_yield.describe())

    # Диаграмма распределения доходности
    print(daily_log_yield.describe())  # Общая статистика quarter = {DataFrame: (2, 6)} Open        High  ...   Adj Close        Volume [Date                                ...                          ] [2020-11-30  126.311429  131.634287  ...  123.875861  1.196460e+07] [2021-03-31  139.216551  141.682758  ...  136.010069  8.204011e+06] [] [...View as DataFrame
   # daily_log_yield.hist(bins=50, sharex=True, figsize=(20, 8))  # Распределение
    plt.show()

if __name__ == '__main__':
    menu()