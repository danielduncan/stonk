from flask import Flask, render_template, request, jsonify
import os
from waitress import serve

app = Flask(__name__)

@app.route('/')
# how do names work and how do i pass a variable through to the bot
def index():
    import data
    return render_template('index.html', price1 = data.mktPrice('AMD'), price2 = data.mktPrice('NIO'), price3 = data.mktPrice('DRO.AX'), price4 = data.mktPrice('PLUG'))

@app.route('/trade', methods=['GET'])
def trade():
    return render_template('render.html')

@app.route('/trading', methods=['POST'])
def trading():
    if request.form['ticker'].isalpha() == True:
        import analysis
        return str(analysis.analysis(request.form['ticker']))
    else:
        return 'Not a ticker!'

# checks if user is connected to Interactive Brokers
@app.route('/connectivity')
def connectivity():
    import connectivitytest
    return "Check console for connection status."

# serve(app, host='0.0.0.0', port=8080, threads=1) # waitress