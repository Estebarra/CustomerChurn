import os
import logging
import glob
import sys 
import pytest
import joblib

import constants
import churn_library as cl


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture(name='df')
def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the 
    other test functions
    '''
    try:
        df = cl.import_data("./data/bank_data.csv")
        logging.info("Importing data was succesful!")
    except FileNotFoundError as err:
        logging.error("I found an error, I didn't find the file")
        raise err
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("The imported dataframe has a reasonable shape!")
    except AssertionError as err:
        logging.error("The imported dataframe has a strange shape")
        raise err

    return df

def test_eda(df):
    '''
    test perform eda function
    '''
    cl.perform_eda(df)
    for image_name in constants.EDA_GRAPHS:
        try:
            with open("images/eda/%s.png" % image_name, 'r'):
                logging.info("I found {}!".format(image_name))
        except FileNotFoundError as err:
            logging.error("I did not found {}".format(image_name))
            raise err

@pytest.fixture(name='df_encoded')
def test_encoder_helper(df):
    '''
    test encoder helper
    '''
    columns = [i + '_' + constants.RESPONSE for i in constants.CAT_COLUMNS]
    try:
        df_encoded = cl.encoder_helper(df, constants.CAT_COLUMNS)
        logging.info("Encoded dataframe fixture creation: SUCCESS")
    except KeyError as err:
        logging.error(
            " Not existent column to encode")
        raise err
    try:
        for column in columns:
            assert column in df_encoded
    except AssertionError as err:
        logging.error(
            "The dataframe doesn't have the right encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")

    return df_encoded

@pytest.fixture(name='data')
def test_perform_feature_engineering(df_encoded):
    '''
    test perform_feature_engineering
    '''
    data = dict()
    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            df_encoded)

        logging.info("Feature sequence fixture creation: SUCCESS")
    except BaseException:
        logging.error(
            "Feature sequences fixture creation: Sequences length mismatch")
        raise
    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Sequences length mismatch")
        raise err

    data["x_train"] = x_train
    data["x_test"] = x_test
    data["y_train"] = y_train
    data["y_test"] = y_test

    return data


def test_train_models(data):
    '''
    test train_models
    '''
    cl.train_models(
        data["X_train"],
        data["X_test"],
        data['y_test'],
        data['y_train'])
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing testing_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The files waeren't found")
        raise err
    for image_name in [
        "Logistic_Regression",
        "Random_Forest",
        "Feature_Importance"]:
        try:
            with open("images/results/%s.png" % image_name, 'r'):
                logging.info("SUCCESS")
        except FileNotFoundError as err:
            logging.error("generated images missing")
            raise err


if __name__ == "__main__":
    for directory in ["logs", "images/eda", "images/results", "./models"]:
        files = glob.glob("%s/*" % directory)
        for file in files:
            os.remove(file)
    sys.exit(pytest.main(["-s"]))








