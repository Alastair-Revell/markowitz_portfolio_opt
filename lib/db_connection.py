import pandas_datareader as web
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import psycopg2

class DatabaseConnection:
    def __init__(self):
        try:
            self.connection = psycopg2.connect(
                "dbname='week4_optimizer' user='student' host='localhost' port='5432'")
            self.connection.autocommit = True
            self.cursor = self.connection.cursor()
        except:
            print('Cannot connect to database')

    def get_price_data(self):
        self.cursor.execute("SELECT * FROM price_data;")
        data = self.cursor.fetchall()

    def get_returns_data(self, tickers):
        self.cursor.execute("select column_name from information_schema.columns where table_name = price_data
        and ordinal_position = 2;")
        data = self.cursor.fetchall()

if __name__ == '__main__':

    def get_ticker():
        ticker = input("Please enter the ticker: ").upper()
        return ticker


    def get_data(ticker):

        start = datetime(2016, 9, 1)
        end   = datetime(2019, 2, 1)

        f = web.DataReader(ticker, 'iex', start, end)
        df = f["close"]
        return df


    database_connection = DatabaseConnection()
    database_connection.get_returns_data()
