from flask import Flask
from flask import request
from flask import render_template,url_for
import phoenixModel
app = Flask(__name__)

@app.route('/')
def index(name="TreeHouse"):
	return render_template('index.html')


@app.route('/calculate/',methods=['POST'])
def calculate():
	reqDict = request.form
	age = int(reqDict['Age'])
	sp = reqDict['sp']
	ss = reqDict['ss']
	pt = reqDict['pt']
	inr = float(reqDict['inr'])
	plt = float(reqDict['plt'])
	hemo = float(reqDict['hemo'])
	retValue = phoenixModel.mainModel(age,sp,ss,pt,inr,plt,hemo)
	print(retValue)
	return render_template('index.html',rbc = retValue)
	

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

app.run(debug=True,host='0.0.0.0')

