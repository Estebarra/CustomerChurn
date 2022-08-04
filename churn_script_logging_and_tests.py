'''
TO DO
'''
import os
import logging
import glob
import sys
import pytest
import joblib

import constants
import churn_library as cl

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='./logs/testing_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

test_model = cl.ChurnModel()


def test_import():
    '''
    test data import - this example is completed for you to assist with the
    other test functions
    '''
    try:
        test_df = test_model.import_data("./data/bank_data.csv")
        logging.info("Importing data was succesful!")
    except FileNotFoundError as err:
        logging.error("I found an error, I didn't find the file")
        raise err
    try:
        assert test_df.shape[0] > 0
        assert test_df.shape[1] > 0
        logging.info("The imported dataframe has a reasonable shape!")
    except AssertionError as err:
        logging.error("The imported dataframe has a strange shape")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    test_model.perform_eda()
    for image_name in constants.EDA_GRAPHS:
        try:
            with open(f"images/eda/{image_name}.png", 'r', encoding='utf-8'):
                logging.info("I found %s!", image_name)
        except FileNotFoundError as err:
            logging.error("I did not found %s", image_name)
            raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    columns = [i + '_' + constants.RESPONSE for i in constants.CAT_COLUMNS]
    try:
        df_encoded = test_model.encoder_helper(constants.CAT_COLUMNS,
                                               constants.RESPONSE)
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


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = test_model.perform_feature_engineering()

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


def test_train_models():
    '''
    test train_models
    '''
    test_model.train_models()
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing testing_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The files waeren't found")
        raise err
    for image_name in [
        "feature_importances",
        "logistic_results",
        "rf_results",
        "roc_curve_result",
            "shap_plot"]:
        try:
            with open(f"images/results/{image_name}.png", 'r', encoding='utf-8'):
                logging.info("SUCCESS")
        except FileNotFoundError as err:
            logging.error("generated images missing")
            raise err


if __name__ == "__main__":
    for directory in ["logs", "images/eda", "images/results", "./models"]:
        files = glob.glob(f"{directory}/*")
        for file in files:
            os.remove(file)
    sys.exit(pytest.main(["-s"]))
