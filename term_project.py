import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

plt.rcParams['figure.figsize'] = (20, 10)

from pandas_datareader.data import DataReader
from datetime import datetime


# INPUT #######                ## Tum proje icin kullanabilegemiz global variablelarimiz.
TECH_LIST = ['TSLA']          ## Buraya istedigimiz tickeri yazabiliriz.
TECH_NAMES = ['TESLA']
MOVING_AVERAGE_DAYS = [10, 20, 30, 50]
END = datetime.now()
START = datetime(END.year - 2, END.month, END.day)

##############################

TICKER_TO_STOCK = {}      ## Bir key istedgimiz zaman suslu parantezle keyin degerini donuyor.Koseli yapsaydik parantez liste olacakti, suslu parantezle dictionary haline getirdik.
TICKER_TO_NAME = {}

def set_globals():
    global TICKER_TO_STOCK
    global TICKER_TO_NAME

    for ticker in TECH_LIST:
        ticker_stock_data = DataReader(ticker, 'yahoo', START, END)
        TICKER_TO_STOCK[ticker] = ticker_stock_data

    for ticker, name in zip (TECH_LIST, TECH_NAMES):
        TICKER_TO_NAME[ticker] = name

    for ticker in TICKER_TO_STOCK:
        stock = TICKER_TO_STOCK[ticker]
        stock["company_name"] = TICKER_TO_NAME[ticker]
        TICKER_TO_STOCK[ticker] = stock

def get_summary_stats():
    df = pd.concat(TICKER_TO_STOCK.values(), axis=0)   ##Tum stocklari unionlayip, son 10 rowu aldik.
    print('---------------SUMMARY---------------')
    print(df.tail(10))

def get_stock_summaries():
    for ticker in TICKER_TO_STOCK:
        name = TICKER_TO_NAME[ticker]
        print('----------------' + name + ' STATS----------------')

        stock_data = TICKER_TO_STOCK[ticker]
        print(stock_data.describe())

def show_historical_closing_price():

    for ticker in TICKER_TO_STOCK:
        plt.figure()
        stock_data = TICKER_TO_STOCK[ticker]
        stock_data['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(ticker)
        plt.show()

def show_volume_history():
    for ticker in TICKER_TO_STOCK:
        plt.figure()
        stock_data = TICKER_TO_STOCK[ticker]
        stock_data['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title(ticker)
        plt.show()

def get_moving_averages():
    column_names_to_be_plot = ['Adj Close']   ##her bir movig averagei ayni plota basicaz o yuzden isimleri belirlicez.
    for moving_average_day in MOVING_AVERAGE_DAYS:
        for ticker in TICKER_TO_STOCK:
            stock_data = TICKER_TO_STOCK[ticker]
            moving_average_column_name = 'MA-' + str(moving_average_day)
            stock_data[moving_average_column_name] = stock_data['Adj Close'].rolling(moving_average_day).mean()  ##stock datamiza moving average columnni eklicez.
        column_names_to_be_plot.append(moving_average_column_name)  ##plot ederken direkt o columnu cekmek icin bunu ekledik. asagidaki plotta yazdik bunu.


    for ticker in TICKER_TO_STOCK:
        stock_data = TICKER_TO_STOCK[ticker]

        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)  #her bir figurede 1 tane subplot yani 1 plot yazmak icin bunu boyle olusturduk.
        fig.set_figheight(8)
        fig.set_figwidth(15)
        stock_data[column_names_to_be_plot].plot(ax=axes[0,0])
        axes[0,0].set_title(TICKER_TO_NAME[ticker])
        fig.show()


def get_macd():
    for ticker in TICKER_TO_STOCK:
        stock_data = TICKER_TO_STOCK[ticker]['Adj Close']
        exp1 = stock_data.ewm(span=12, adjust=False).mean()
        exp2 = stock_data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        exp3 = macd.ewm(span=9, adjust=False).mean()
        macd.plot(label=ticker + ' MACD', color='g')
        ax = exp3.plot(label='Signal Line', color='r')
        stock_data.plot(ax=ax, secondary_y=True, label=ticker)

        ax.set_ylabel('MACD')
        ax.right_ax.set_ylabel('Price $')
        ax.set_xlabel('Date')
        lines = ax.get_lines() + ax.right_ax.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc='upper left')
        plt.show()

def get_rsi(time_window): #https://tcoil.info/compute-rsi-for-stocks-with-python-relative-strength-index/
    for ticker in TICKER_TO_STOCK:
        stock_data = TICKER_TO_STOCK[ticker]['Adj Close']
        diff = stock_data.diff(1).dropna() #close fiyatlarinin farklarini al

        up_change = 0 * diff
        down_change = 0 * diff

        up_change[diff > 0] = diff[diff > 0]
        down_change[diff < 0] = diff[diff < 0]

        # check pandas documentation for ewm
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
        # values are related to exponential decay
        # we set com=time_window-1 so we get decay alpha=1/time_window
        up_change_average = up_change.ewm(com=time_window - 1, min_periods=time_window).mean()
        down_change_average = down_change.ewm(com=time_window - 1, min_periods=time_window).mean()

        rs = abs(up_change_average/down_change_average)
        rsi = 100 - 100/(1 + rs)

        #close price plotunu ciz
        plt.figure()
        plt.plot(TICKER_TO_STOCK[ticker]['Adj Close'])
        plt.title('Price Chart (Adj Close): ' + TICKER_TO_NAME[ticker])
        plt.show()

        #rsi ve significant levellari ciz
        plt.figure()
        plt.title('RSI chart: ' + TICKER_TO_NAME[ticker])
        plt.plot(rsi)

        plt.axhline(0, linestyle='--', alpha=0.1)
        plt.axhline(20, linestyle='--', alpha=0.5)
        plt.axhline(30, linestyle='--')

        plt.axhline(70, linestyle='--')
        plt.axhline(80, linestyle='--', alpha=0.5)
        plt.axhline(100, linestyle='--', alpha=0.1)
        plt.show()

def get_psar():#https://blog.quantinsti.com/parabolic-sar/
    for ticker in TICKER_TO_STOCK:
        stock_data = TICKER_TO_STOCK[ticker]
        stock_data.ta.psar(append=True)
        stock_data[['adj_close', 'PSARs_0.02_0.2', 'PSARl_0.02_0.2']].plot()
        plt.title('Parabolic SAR: ' + TICKER_TO_NAME[ticker])
        plt.grid()
        plt.show()

def get_stochastic_oscillator():
    for ticker in TICKER_TO_STOCK:
        stock_data = TICKER_TO_STOCK[ticker]
        stock_data['14-High'] = stock_data['High'].rolling(14).max()
        stock_data['14-Low'] = stock_data['Low'].rolling(14).min()
        stock_data['%K'] = (stock_data['Adj Close'] - stock_data['14-Low'])*100/(stock_data['14-High'] - stock_data['14-Low'])
        stock_data['%D'] = stock_data['%K'].rolling(3).mean()


        ax = stock_data[['%K', '%D']].plot()
        stock_data['Adj Close'].plot(ax=ax, secondary_y=True)
        ax.axhline(20, linestyle='--', color="r")
        ax.axhline(80, linestyle="--", color="r")
        plt.title('Stochastic Oscillator: ' + TICKER_TO_NAME[ticker])
        plt.show(dpi=1200)

def get_cci():#https://www.exfinsis.com/tutorials/python-programming-language/cci-stock-technical-indicator-with-python/
    for ticker in TICKER_TO_STOCK:
        stock_data = TICKER_TO_STOCK[ticker]
        stock_data.ta.cci(length=20, append=True)
        stock_data[['adj_close', 'CCI_20_0.015']].plot()
        plt.title('CCI(20, 0.015): ' + TICKER_TO_NAME[ticker])
        plt.grid()


        plt.axhline(150, linestyle='--', alpha=0.5)
        plt.axhline(-150, linestyle='--', alpha=0.5)
        plt.show()

def get_awesome_oscillator(fast, slow):
    for ticker in TICKER_TO_STOCK:
        stock_data = TICKER_TO_STOCK[ticker]
        stock_data.ta.ao(fast=fast, slow=slow, append=True)
        # stock_data[['adj_close', 'AO_' + str(fast) + '_' + str(slow)]].plot()
        # plt.title('Awesome Oscillator_' + str(fast) + '_' + str(slow) + ': ' + TICKER_TO_NAME[ticker])
        # plt.grid()
        #
        # plt.show()

        ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 5, colspan = 1)
        ax2 = plt.subplot2grid((10,1), (6,0), rowspan = 4, colspan = 1)
        ax1.plot(stock_data['adj_close'])
        ax1.set_title(TICKER_TO_NAME[ticker] + ' CLOSING PRICE')
        for i in range(len(stock_data)):
            if stock_data['AO_' + str(fast) + '_' + str(slow)][i-1] > stock_data['AO_' + str(fast) + '_' + str(slow)][i]:
                ax2.bar(stock_data.index[i], stock_data['AO_' + str(fast) + '_' + str(slow)][i], color = '#f44336')
            else:
                ax2.bar(stock_data.index[i], stock_data['AO_' + str(fast) + '_' + str(slow)][i], color = '#26a69a')
        ax2.set_title('Awesome Oscillator_' + str(fast) + '_' + str(slow) + ': ' + TICKER_TO_NAME[ticker])
        plt.show()

set_globals()
#get_summary_stats()
#get_stock_summaries()
# show_historical_closing_price()
# show_volume_history()
# get_moving_averages()
# get_macd()
# get_rsi(14)
# get_psar()
get_stochastic_oscillator()
# get_cci()
# get_awesome_oscillator(fast=5, slow=34)