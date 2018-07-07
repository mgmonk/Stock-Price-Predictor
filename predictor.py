import csv
# import numpy as np
# from sklearn.svm import SVR
# import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[1:]))
            prices.append(float(row[4]))
    return

get_data('AAPL.csv')

print(dates[0:10])
print(prices[0:10])

def predict_price(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernal = 'linear', C=1e3)
    svr_poly = SVR(kernal = 'polynomial', C=1e3, degree=2)
    svr_rbf = SVR(kernal = 'rbf', C=1e3, gamma = 0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
