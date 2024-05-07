from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

# required conversion for the model
diagnosis_dict = {
    0: "Nhi Hai",
    1: "Hai"
}

@app.route("/")   # used to define the routes on the web pages
def index_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET","POST"])
def predictPage():
    float_features = [float(x) for x in request.form.values()]
    mean = [14.12729174,19.28964851,91.96903339,654.8891037,0.096360281,0.104340984,0.088799316,0.048919146,0.181161863,0.06279761,0.405172056,1.216853427,2.866059227,40.33707909,0.007040979,0.025478139,0.031893716,0.011796137,0.020542299,0.003794904,16.26918981,25.6772232,107.2612127,880.5831283,0.132368594,0.254265044,0.272188483,0.114606223,0.290075571,0.083945817]
    std_dev = [3.524048826,4.301035768,24.29898104,351.9141292,0.014064128,0.052812758,0.079719809,0.038802845,0.027414281,0.007060363,0.277312733,0.551648393,2.021854554,45.49100552,0.003002518,0.017908179,0.03018606,0.006170285,0.008266372,0.002646071,4.83324158,6.146257623,33.60254227,569.3569927,0.022832429,0.157336489,0.208624281,0.065732341,0.061867468,0.018061267]
    reverse_scaled_features = []
    for i in range(len(float_features)):
        reverse_scaled_features.append((float_features[i] - mean[i])/std_dev[i])
    features = [np.array(reverse_scaled_features)]
    predictedFinalClass = model.predict(features)

    predictedFinalClass = diagnosis_dict[predictedFinalClass[0]]    

    return render_template("index.html", prediction = predictedFinalClass)


if __name__ == "__main__":
    app.run(debug=True)  # by writing true it automatically detects the changes


