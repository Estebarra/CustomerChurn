# library doc string
"""
To do
"""

import logging
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

sns.set()
plt.rcParams['figure.figsize'] = 20, 10


def plot_hist(churn_df, column, name):
    '''
    Function
    '''
    plt.figure()
    churn_df[column].hist()
    plt.savefig('images/eda/' + name)
    plt.clf()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest
    output:
        None
    '''
    logging.info('Initializing classification report creation...')
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01,
             1.25,
             str('Random Forest Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.6,
             str('Random Forest Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/rf_results.png')
    plt.clf()
    logging.info('RF report created and stored succesfully')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01,
             1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/logistic_results.png')
    plt.clf()
    logging.info('LR report created and stored succesfully')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth

    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure
    output:
        None
    '''
    logging.info('Initializing feature importance plot creation')
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title('Feature Importance')
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth + 'feature_importances.png')
    plt.clf()
    logging.info('Feature importance plot created and stored succesfully')


class ChurnModel:
    '''
    Class
    '''

    def __init__(self):
        self.churn_df = None
        self.x_train = None
        self.x_test = None
        self.y_test = None
        self.y_train = None

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth

        input:
            pth: a path to the csv
        output:
            df: pandas dataframe
        '''
        logging.info('Importing data...')
        try:
            self.churn_df = pd.read_csv(pth)

            self.churn_df['Churn'] = self.churn_df['Attrition_Flag']\
                .apply(lambda val: 0 if val == "Existing Customer" else 1)

            logging.info('Data imported succesfully')

            logging.info('Data has a shape of: %s', self.churn_df.shape)

            return self.churn_df

        except FileNotFoundError:
            logging.error('The file was not available')

    def perform_eda(self):
        '''
        perform eda on df and save figures to images folder

        input:
            df: pandas dataframe
        output:
            None
        '''
        logging.info('Initializing EDA')
        plot_hist(self.churn_df,
                  'Churn',
                  'churn_distribution.png')
        logging.info('Churn distribution plot created and stored succesfully')
        plot_hist(self.churn_df,
                  'Customer_Age',
                  'customer_age_distribution.png')
        logging.info(
            'Customer Age distribution plot created and stored succesfully')
        plt.figure()
        self.churn_df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig('images/eda/marital_status_distribution.png')
        plt.clf()
        logging.info(
            'Marital Status distribution plot created and stored succesfully')

        plt.figure()
        sns.distplot(self.churn_df['Total_Trans_Ct'])
        plt.savefig('images/eda/total_transaction_distribution.png')
        plt.clf()
        logging.info(
            'Total Transaction distribution plot created and stored succesfully')

        plt.figure()
        sns.heatmap(self.churn_df.corr(),
                    annot=False,
                    cmap='Dark2_r',
                    linewidths=2)
        plt.savefig('images/eda/heatmap.png')
        plt.clf()
        logging.info('Heatmap created and stored succesfully')

    def encoder_helper(self, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the
        notebook

        input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                used for naming variables or index y column]
        output:
            df: pandas dataframe with new columns for
        '''
        logging.info('Initializing Encoder Helper')
        for i in category_lst:
            cat_lst = []
            cat_groups = self.churn_df.groupby(i).mean()['Churn']

            for val in self.churn_df[i]:
                cat_lst.append(cat_groups.loc[val])

            self.churn_df[i + '_' + response] = cat_lst

        logging.info('Encoder Helper executed succesfully')
        logging.info('Dataframe has a new shape of: %s', self.churn_df.shape)

        return self.churn_df

    def perform_feature_engineering(self):
        '''
        input:
            df: pandas dataframe
            response: string of response name [optional argument that could be
            used for naming variables or index y column]
        output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
        '''
        logging.info('Initializing Feature Engineering')
        features = pd.DataFrame()
        features[constants.KEEP_COLS] = self.churn_df[constants.KEEP_COLS]

        labels = self.churn_df['Churn']

        self.x_train, self.x_test, self.y_train, self.y_test =\
            train_test_split(features,
                             labels,
                             test_size=0.3,
                             random_state=42)
        logging.info("Feature Engineering executed succesfully")
        logging.info("X_train has a shape of: %s", self.x_train.shape)
        logging.info("X_test has a shape of: %s", self.x_test.shape)
        logging.info("y_train has a shape of: %s", self.y_train.shape)
        logging.info("y_test has a shape of: %s", self.y_test.shape)

        return self.x_train, self.x_test, self.y_train, self.y_test

    def train_models(self):
        '''
        train, store model results: images + scores, and store models

        input:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
        output:
            None
        '''
        logging.info('Initializing Training')
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()

        cv_rfc = GridSearchCV(estimator=rfc,
                              param_grid=constants.PARAM_GRID,
                              cv=5)
        cv_rfc.fit(self.x_train, self.y_train)

        lrc.fit(self.x_train, self.y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.x_test)

        y_train_preds_lr = lrc.predict(self.x_train)
        y_test_preds_lr = lrc.predict(self.x_test)

        # Save best models
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        logging.info('RF Model has been succesfully stored')
        joblib.dump(lrc, './models/logistic_model.pkl')
        logging.info('LR Model has been succesfully stored')

        # Load models
        rfc_model = joblib.load('./models/rfc_model.pkl')
        logging.info('RF Model has been succesfully loaded')
        lr_model = joblib.load('./models/logistic_model.pkl')
        logging.info('LR Model has been succesfully loaded')

        classification_report_image(self.y_train,
                                    self.y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf)

        x_data = pd.concat([self.x_test, self.x_train], ignore_index=True)

        feature_importance_plot(rfc_model, x_data, './images/results/')

        # Plot and save ROC Curve
        lrc_plot = plot_roc_curve(lr_model, self.x_test, self.y_test)
        plt.figure(figsize=(15, 8))
        axes = plt.gca()
        _ = plot_roc_curve(rfc_model,
                           self.x_test,
                           self.y_test,
                           ax=axes,
                           alpha=0.8)
        lrc_plot.plot(ax=axes, alpha=0.8)
        plt.savefig('./images/results/roc_curve_result.png')
        plt.clf()
        logging.info('ROC curve plot has been succesfully created and stored')

        explainer = shap.TreeExplainer(rfc_model)
        shap_values = explainer.shap_values(self.x_test)
        shap.summary_plot(shap_values, self.x_test, plot_type="bar")
        plt.savefig('./images/results/shap_plot.png')
        plt.clf()
        logging.info('SHAP plot has been succesfully created and stored')


if __name__ == "__main__":

    # model object initiation
    MODEL_INS = ChurnModel()

    # read the data
    MODEL_INS.import_data(constants.DATA_PATH)

    # create eda plot and save the result in images/eda
    MODEL_INS.perform_eda()

    # encoding categorical feature
    MODEL_INS.encoder_helper(constants.CAT_COLUMNS, constants.RESPONSE)

    # feature engineering (standardization and data splitting)
    MODEL_INS.perform_feature_engineering()

    # model training and evaluation
    # model object was saved with .pkl extension in models folder
    # model evaluation result was saved in images/results
    MODEL_INS.train_models()
