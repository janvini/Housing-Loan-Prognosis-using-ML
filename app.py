# save this as app.py
from flask import Flask, escape, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('Rfmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")





@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method ==  'POST':
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employed = request.form['employed']
        credit = request.form['credit']
        prop_area = request.form['area']
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])

        # gender
        if (gender == "Male"):
            male=1
        else:
            male=0
        
        # married
        if(married=="Yes"):
            married_yes = 1
        else:
            married_yes=0

        # dependents
        if dependents == '0':
            dependents = 0
        elif dependents == '1':
            dependents = 1
        elif dependents == '2':
            dependents = 2
        else:
            dependents = 3 

        # education
        if (education=="Not Graduate"):
            not_graduate=1
        else:
            not_graduate=0
 
        # employed
        if (employed == "Yes"):
            employed_yes=1
        else:
            employed_yes=0

        if (credit == "yes"):
            credit_his =1
        else:
            credit_his=0

        # property area

        if prop_area == 'Rural':
            prop_area = 0
        elif prop_area == 'Semiurban':
            prop_area = 1
        else:
            prop_area = 2


        

        prediction = model.predict([[male, married_yes, dependents, not_graduate, employed_yes, ApplicantIncome,CoapplicantIncome,LoanAmount, Loan_Amount_Term, credit_his,prop_area ]])

        # print(prediction)

        if(prediction==[0]):
            prediction="No"
        else:
            prediction="Yes"


        return render_template("prediction.html", prediction_text="loan status is {}".format(prediction))




    else:
        return render_template("prediction.html")



if __name__ == "__main__":
    app.run(debug=True)