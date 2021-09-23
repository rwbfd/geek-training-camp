import time
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_docker():
	return "Hello Docker!\n"

@app.route('/healths')
def health_success():
	return "OK"

@app.route('/healthf')
def health_fail():
	time.sleep(10)
	return 'OK'

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
