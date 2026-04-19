# Server file to run the trading environment

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Trading Server Running!'

if __name__ == '__main__':
    app.run(debug=True)
