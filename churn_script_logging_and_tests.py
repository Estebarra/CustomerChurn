"""
This file contains the tests used to check the workflow functionality
Author: Luis Barranco
Date: August 05, 2022
"""

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
    filename='./logs/churn_library.log',
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
        test_df = test_model.import_data('./data/bank_data.csv')
        logging.info('TESTING: Importing data was succesful!')
    except FileNotFoundError as err:
        logging.error('TESTING: I found an error, I did not find the file')
        raise err
    try:
        assert test_df.shape[0] > 0
        assert test_df.shape[1] > 0
        logging.info('TESTING: The imported dataframe has a reasonable shape!')
    except AssertionError as err:
        logging.error('TESTING: The imported dataframe has a strange shape')
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    test_model.perform_eda()
    for image_name in constants.EDA_GRAPHS:
        try:
            with open(f'images/eda/{image_name}.png', 'r', encoding='utf-8'):
                logging.info('TESTING: I found %s!', image_name)
        except FileNotFoundError as err:
            logging.error('TESTING: I did not found %s', image_name)
            raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    columns = [i + '_' + constants.RESPONSE for i in constants.CAT_COLUMNS]
    try:
        df_encoded = test_model.encoder_helper(constants.CAT_COLUMNS,
                                               constants.RESPONSE)
        logging.info('TESTING: Encoded dataframe created!')
    except KeyError as err:
        logging.error(
            'TESTING: Not existent column to encode')
        raise err
    try:
        for column in columns:
            assert column in df_encoded
        logging.info('TESTING: The dataframe has the right encoded columns!')
    except AssertionError as err:
        logging.error(
            'TESTING: The dataframe does not have the right encoded columns')
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = test_model.perform_feature_engineering()

        logging.info('TESTING: Training and testing sets created!')
    except BaseException:
        logging.error(
            'TESTING: Training and testing sets were not created')
        raise
    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info(
            'TESTING: Test and training sets have the correct dimensions!')
    except AssertionError as err:
        logging.error('TESTING: Test and training sets length mismatch')
        raise err


def test_train_models():
    '''
    test train_models
    '''
    test_model.train_models()
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info('TESTING: Models Trained!')
    except FileNotFoundError as err:
        logging.error('TESTING: The models files were not found')
        raise err
    for image_name in [
        'feature_importances',
        'logistic_results',
        'rf_results',
        'roc_curve_result',
            'shap_plot']:
        try:
            with open(f'images/results/{image_name}.png', 'r', encoding='utf-8'):
                logging.info('TESTING: Image found!')
        except FileNotFoundError as err:
            logging.error('TESTING: Generated images missing')
            raise err


if __name__ == '__main__':
    for directory in ['logs', 'images/eda', 'images/results', './models']:
        files = glob.glob(f'{directory}/*')
        for file in files:
            os.remove(file)
    sys.exit(pytest.main(['-s']))
