from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')

# dump information to that file
clf = pickle.load(file)

# close the file
file.close()


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        Children = float(myDict['Children'])
        Cases1M = float(myDict['Cases/1M'])
        Deaths1M = float(myDict['Deaths/1M'])
        Age = float(myDict['Age'])
        Comascore = float(myDict['Comascore'])
        Diuresis = float(myDict['Diuresis'])
        Platelets = float(myDict['Platelets'])
        HBB = float(myDict['HBB'])
        ddimer = float(myDict['ddimer'])
        Heartrate = float(myDict['Heartrate'])
        HDLcholesterol = float(myDict['HDLcholesterol'])
        Charlsonindex = float(myDict['Charlsonindex'])
        Bloodglucose = float(myDict['Bloodglucose'])
        Insurance = float(myDict['Insurance'])
        Salary = float(myDict['Salary'])
        FTmonth = float(myDict['FT/month'])
        print(request.form)
        inputFeatures = [Children, Cases1M, Deaths1M, Age, Comascore, Diuresis, Platelets, HBB, ddimer, Heartrate,
                         HDLcholesterol, Charlsonindex, Bloodglucose, Insurance, Salary, FTmonth]
        InfProb = clf.predict_proba([inputFeatures])[0][0]
        print(InfProb)
        return render_template('show.html', inf=round(InfProb*100))
    return render_template('index.html')
        # return 'Hello, Patient!' + str(InfProb)


if __name__ == "__main__":
    app.run(debug=True)
