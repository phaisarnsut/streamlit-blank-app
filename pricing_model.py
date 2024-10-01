# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:28:10 2022

@author: maxim migutin
"""

from datetime import date
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib
from loguru import logger

import warnings
warnings.filterwarnings("ignore")

def main():
    logger.info("Modules loaded")

    df_main = pd.read_csv('cars.csv')
    logger.info("Data loaded")
    
    # Data preparation
    df_main['Age'] = date.today().year - df_main['Year']
    df_main.drop('Year', axis=1, inplace = True)
    df_main.rename(columns = {'Selling_Price':'Selling_Price(lacs)','Present_Price':'Present_Price(lacs)','Owner':'Past_Owners'},inplace = True)
    df_main.drop(labels='Car_Name',axis= 1, inplace = True)
    enc = OneHotEncoder(handle_unknown='ignore')

    X = df_main.drop(['Selling_Price(lacs)', 'Present_Price(lacs)'], axis=1)
    enc.fit(X)
    X = enc.transform(X)

    y = df_main['Selling_Price(lacs)']

    with open('encoder.sav', 'wb') as f:
        joblib.dump(enc, f)

    logger.info("Data processed, OHE persisted")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4217)

    rf = RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=5, min_samples_split=7, max_features='sqrt')
    rf.fit(X_train, y_train)

    logger.info("Model trained")

    pred=rf.predict(X_test)
    logger.info("MAPE score achieved: {}".format(mean_absolute_percentage_error(y_test, pred)))

    model = rf.fit(X, y)

    joblib.dump(model, "regressor.sav")
    logger.info("ML model dumped into binary")

if __name__=='__main__':
    main()