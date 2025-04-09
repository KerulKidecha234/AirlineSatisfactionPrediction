from flask import Flask, render_template, request

app = Flask(__name__)


def predict(data):

    return all(int(value) >= 4 for value in data.values())

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_page():

    form_data = request.form.to_dict()


    satisfied = predict(form_data)

    return render_template('after.html', satisfied=satisfied)

if __name__ == '__main__':
    app.run(debug=True)

