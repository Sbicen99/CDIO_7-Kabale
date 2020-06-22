import json

import CardDetector

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello1():
    return 'Prøv "/data" i stedet!'


@app.route('/data')
def hello2():
    try:
        with open('kabalen2.json', 'r') as file:
            return file.read()
    except:
        return '... Løbet tør for data, hvor trist!'

@app.route('/data2')
def test():
    CardDetector.startMainLoop()
    try:
        with open('kabalen2.json', 'r') as file:
            return file.read()
    except:
        return '... Løbet tør for data, hvor trist!'


if __name__ == '__main__':
    app.run()
