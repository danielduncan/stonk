from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return render_template(render.html)

@app.route('/home')
    

@app.route('/working')
def working():
    return 'Working!'