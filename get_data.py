# alter it to work with any list of securities
# discover more about quadratic programming
# explain the optimisation process in more detail
# map out routes for web app
# add delete and add buttons for web app
# look at rendering graphs on a web app

import pandas_datareader as web
from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import psycopg2
import cvxopt as opt
from cvxopt import blas, solvers
import numpy as np
import matplotlib.dates as mdates
import pygal
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
import json
import mpld3


solvers.options['show_progress'] = False
returns_table = pd.read_csv("returns_data.csv", sep=',')
normedret_table = pd.read_csv("normed_ret.csv", sep=',')
price_table = pd.read_csv("price_data.csv", sep=',')

stocks = ['MSFT','AAPL','AMZN']

def get_ticker():
    ticker = input("Please enter the ticker: ").upper()
    return ticker

def get_data(ticker):

    start = datetime(2016, 9, 1)
    end   = datetime(2019, 2, 1)

    f = web.DataReader(ticker, 'iex', start, end)
    df = f["close"]
    print(df.head())
    return df

def get_data_and_append():
    ticker = get_ticker()
    data = (get_data(ticker))
    dates['{}'.format(ticker)] = (get_data(ticker)).tolist()

return_vec = (((returns_table[stocks]).values).T)

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

n_portfolios = 2000
means, stds = np.column_stack([
    random_portfolio(return_vec)
    for _ in range(n_portfolios)
])

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    print(m1[2])
    print(m1[0])
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


def print_optimization(stocks):

    weights, returns, risks = optimal_portfolio((((returns_table[stocks]).values).T))

    print(weights)

    n_portfolios = 2000
    means, stds = np.column_stack([
        random_portfolio((((returns_table[stocks]).values).T))
        for _ in range(n_portfolios)
    ])

    plt.plot(stds, means, 'o', markersize=2)
    plt.ylabel('mean')
    plt.xlabel('std')
    plt.plot(risks, returns, 'y-o', markersize=2)
    plt.show()

def print_stock_returns(stocks):

    fig = go.Figure()
    for i in stocks:
        fig.add_scatter(x=returns_table['date'], y=returns_table[i], name=i, mode='lines')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def calculate_portfolio_returns(stocks):

    #Change this later for a selected portfolio
    weights, returns, risks = optimal_portfolio((((returns_table[stocks]).values).T))
    portfolio = stocks
    equal_weights = [1 / len(portfolio)] * len(portfolio)
    optimal_weights = weights.flatten()
    print("The equal weighting: {}".format(equal_weights))
    print("The optimal weighting: {}".format(optimal_weights))
    weighted_stock_return = (normedret_table[stocks].multiply(optimal_weights,axis=1))
    weighted_portfolio_return = weighted_stock_return.sum(axis=1)
    cum_portfolio_return = ((weighted_portfolio_return[1] + 1).cumprod()) - 1

    data = [
        go.Scatter(
            x=price_table['date'], # assign x as the dataframe column 'x'
            y=weighted_portfolio_return
            )
        ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def print_stock_prices(stocks):

    fig = go.Figure()
    for i in stocks:
        fig.add_scatter(x=price_table['date'], y=price_table[i], name=i, mode='lines')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
