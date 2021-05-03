from flask import Flask, render_template, request, jsonify
import os
from waitress import serve


app = Flask(__name__)


@app.route('/')
def index():
    import data
    return render_template('index.html', price1=data.mktPrice('AMD'), price2=data.mktPrice('NIO'), price3=data.mktPrice('DRO.AX'), price4=data.mktPrice('PLUG'))


@app.route('/autoTrad', methods=['GET'])
def autoTrad():
    return render_template('autoTrad.html')


@app.route('/manuTrad', methods=['GET'])
def manuTrad():
    return render_template('manuTrad.html')


@app.route('/prediction', methods=['POST'])
def trading():
    # doesn't allow entry of international tickers (syntax of TIC.EX)
    if request.form['ticker'].isalpha() == True:
        if request.form.get('quantum'):
            import quantum
            return 'qc pred' # str(quantum.analysis(request.form['ticker']))
        else:
            import analysis
            return str(analysis.analysis(request.form['ticker'])) # placeholder
    else:
        return 'Not a ticker!'


# checks if user is connected to Interactive Brokers
@app.route('/connectivity')
def connectivity():
    import connectivitytest
    return "Check console for connection status."


serve(app, host='0.0.0.0', port=8080, threads=8) # waitress
