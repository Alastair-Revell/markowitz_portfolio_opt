from flask import Flask,redirect, request, render_template, session
from get_data import *

app = Flask(__name__)

app.secret_key = (os.urandom(16))

@app.route('/')
def home_page(name=None):
    return render_template('home_page.html', name=name)

@app.route('/add_tickers', methods=['GET','POST'])
def add_tickers():
    if request.method == 'GET':
        return render_template('add_tickers.html')
    else:
        session['stocks'] = request.form.getlist('stocks')
        return redirect("http://127.0.0.1:5000/", code=302)

@app.route('/tickers')
def tickers():
    print(session['stocks'])
    return render_template('view_portfolio.html')

@app.route('/portfolio_return')
def p_return():
    graph = calculate_portfolio_returns(session['stocks'])
    return render_template('view_portfolio_returns.html', plot = graph)

@app.route('/stock_return')
def view_s_return():
    print(session['stocks'])
    graph = print_stock_returns(session['stocks'])
    return render_template('view_stock_returns.html', plot=graph)

@app.route('/price_history')
def p_history():
    graph = print_stock_prices(session['stocks'])
    return render_template('view_portfolio_returns.html', plot = graph)

@app.route('/port_opt')
def port_opt():
    return "opt. weights"

@app.route('/efficient_frontier')
def view_frontier():
    return "Here is the frontier"


# def self.select(title):
# result = Bookmark.connect.exec("SELECT url FROM bookmarks WHERE title = '#{title}';")
# output = result[0]['url']
