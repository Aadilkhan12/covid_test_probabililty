import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

if __name__ == "__main__":
    trainData = pd.read_csv("Train_dataset.csv")
    testData = pd.read_csv("Test_dataset.csv")

    trainData['Children'] = trainData['Children'].astype(np.float32)
    trainData['Diuresis'] = trainData['Diuresis'].astype(np.float32)
    trainData['Platelets'] = trainData['Platelets'].astype(np.float32)
    trainData['HBB'] = trainData['HBB'].astype(np.float32)
    trainData['d-dimer'] = trainData['d-dimer'].astype(np.float32)
    trainData['Heart rate'] = trainData['Heart rate'].astype(np.float32)
    trainData['HDL cholesterol'] = trainData['HDL cholesterol'].astype(np.float32)
    trainData['Insurance'] = trainData['Insurance'].astype(np.float32)
    trainData['FT/month'] = trainData['FT/month'].astype(np.float32)
    trainData['Infect_Prob'] = trainData['Infect_Prob'].astype(np.float32)

    trainData['Name'].fillna(trainData.Name.mode(), inplace=True)
    trainData['Children'].fillna(2, inplace=True)
    trainData['Occupation'].fillna(trainData.Occupation.mode(), inplace=True)
    trainData['Mode_transport'].fillna(trainData.Mode_transport.mode(), inplace=True)
    trainData['comorbidity'].fillna(trainData.comorbidity.mode(), inplace=True)
    trainData['cardiological pressure'].fillna("Normal", inplace=True)
    trainData['FT/month'].fillna(1.0, inplace=True)
    trainData['Diuresis'].fillna(trainData.Diuresis.mean(), inplace=True)
    trainData['Platelets'].fillna(trainData.Platelets.mean(), inplace=True)
    trainData['HBB'].fillna(trainData.HBB.mean(), inplace=True)
    trainData['Insurance'].fillna(trainData.Insurance.mean(), inplace=True)
    trainData['d-dimer'].fillna(275.292292, inplace=True)
    trainData['Heart rate'].fillna(74.847392, inplace=True)
    trainData['HDL cholesterol'].fillna(52.632737, inplace=True)

    trainData['FT/month'] = trainData['FT/month'].astype(np.float32)
    trainData['d-dimer'] = trainData['d-dimer'].astype(np.float32)
    trainData['Heart rate'] = trainData['Heart rate'].astype(np.float32)
    trainData['HDL cholesterol'] = trainData['HDL cholesterol'].astype(np.float32)

    X_train = trainData[
        ['Children', 'cases/1M', 'Deaths/1M', 'Age', 'Coma score', 'Diuresis', 'Platelets', 'HBB',
         'd-dimer', 'Heart rate', 'HDL cholesterol', 'Charlson Index', 'Blood Glucose', 'Insurance', 'salary',
         'FT/month']].to_numpy()
    X_test = testData[
        ['Children', 'cases/1M', 'Deaths/1M', 'Age', 'Coma score', 'Diuresis', 'Platelets', 'HBB',
         'd-dimer', 'Heart rate', 'HDL cholesterol', 'Charlson Index', 'Blood Glucose', 'Insurance', 'salary',
         'FT/month']].to_numpy()

    Y_train = trainData[['Infect_Prob']].to_numpy().reshape(10714, )

    for i in range(10714):
        if Y_train[i] >= 50:
            Y_train[i] = 1
        else:
            Y_train[i] = 0

    testData['Infect_Prob'] = np.nan
    testData['Infect_Prob'].fillna(0, inplace=True)
    Y_test = testData[['Infect_Prob']].to_numpy().reshape(14498, )

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()
