from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
# how do names work and how do i pass a variable through to the bot
def index():
    return render_template('render.html')

@app.route('/home', methods=['POST', 'GET'])
def home():
    import trade
    

@app.route('/working')
def working():
    return 'Working!'