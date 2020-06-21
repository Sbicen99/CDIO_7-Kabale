import json

import CardDetector

from flask import Flask, jsonify

app = Flask(__name__)

a_file = open("inputData.txt", "r")
list_of_lists = []
for line in a_file:
    line_list = line.strip().split(',')

    data_set = {"waste": {
        "value": line_list.__getitem__(0),
        "suit": line_list.__getitem__(1)
    },
        "foundations": [
            {
                "value": line_list.__getitem__(2),
                "suit": line_list.__getitem__(3)
            },
            {
                "value": line_list.__getitem__(4),
                "suit": line_list.__getitem__(5)
            },
            {
                "value": line_list.__getitem__(6),
                "suit": line_list.__getitem__(7)
            },
            {
                "value": line_list.__getitem__(8),
                "suit": line_list.__getitem__(9)
            },
            {
                "value": line_list.__getitem__(10),
                "suit": line_list.__getitem__(11)
            },
            {
                "value": line_list.__getitem__(12),
                "suit": line_list.__getitem__(13)
            },
            {
                "value": line_list.__getitem__(14),
                "suit": line_list.__getitem__(15)
            },
        ]}

    json_dump = json.dumps(data_set)
    list_of_lists.append(json_dump)

a_file.close()
list_of_lists.reverse()


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
