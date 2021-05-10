# serving with flask/waitress
from flask import Flask, render_template, request, jsonify
import os
from waitress import serve


app = Flask(__name__)


# dashboard
@app.route('/')
def index():
    import data
    return render_template('index.html', price1=data.mktPrice('AMD'), price2=data.mktPrice('NIO'), price3=data.mktPrice('DRO.AX'), price4=data.mktPrice('PLUG'))


# auto trading
@app.route('/autoTrad', methods=['GET'])
def autoTrad():
    return render_template('autoTrad.html')


# manual trading
@app.route('/manuTrad', methods=['GET'])
def manuTrad():
    return render_template('manuTrad.html')


# output prediction
@app.route('/prediction', methods=['POST'])
def trading():
    # not secure input method
    if request.method == 'POST': # request.form['ticker'].isalpha() == True: # this method doesn't allow entry of international tickers (syntax of TIC.EX)
        if request.form.get('quantum'):
            import quantum
            return 'qc pred' # str(quantum.analysis(request.form['ticker']))
        else:
            import analysis
            return str(analysis.analysis(request.form['ticker'])) # placeholder
    else:
        return 'Not a ticker!'


# help
@app.route('/inputHelp')
def inputHelp():
    return render_template('inputHelp.html')

@app.route('/dashMontHelp')
def dashMontHelp():
    return render_template('dashMontHelp.html')

@app.route('/dashPerfHelp')
def dashPerfHelp():
    return render_template('dashPerfHelp.html')
    

# error handler for error 500
@app.errorhandler(500)
def page_not_found(e):
    return render_template('500.html'), 500

# checks if user is connected to Interactive Brokers
@app.route('/connectivity')
def connectivity():
    import connectivitytest
    return "Check console for connection status."


serve(app, host='0.0.0.0', port=8080, threads=8) # serving with waitress
