# library doc string
"""
To do
"""

# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import logging
import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

sns.set()
plt.rcParams['figure.figsize'] = 20, 10

def plot_hist(df, column, name):
    '''
    '''
    plt.figure() 
    df[column].hist()
    plt.savefig('images/eda/' + name)
    plt.clf()

    return None

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
            fontproperties = 'monospace')
    plt.text(0.01,
            0.05,
            str(classification_report(y_test, y_test_preds_rf)),
            {'fontsize': 10},
            fontproperties = 'monospace') 
    plt.text(0.01,
            0.6,
            str('Random Forest Test'),
            {'fontsize': 10},
            fontproperties = 'monospace')
    plt.text(0.01,
            0.7, 
            str(classification_report(y_train, y_train_preds_rf)), 
            {'fontsize': 10}, 
            fontproperties = 'monospace') 
    plt.axis('off')
    plt.savefig('images/results/rf_results.png')
    plt.clf()
    logging.info('RF report created and stored succesfully')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01,
            1.25, 
            str('Logistic Regression Train'), 
            {'fontsize': 10}, 
            fontproperties = 'monospace')
    plt.text(0.01, 
            0.05, 
            str(classification_report(y_train, y_train_preds_lr)), 
            {'fontsize': 10}, 
            fontproperties = 'monospace')
    plt.text(0.01, 
            0.6, 
            str('Logistic Regression Test'), 
            {'fontsize': 10}, 
            fontproperties = 'monospace')
    plt.text(0.01, 
            0.7, 
            str(classification_report(y_test, y_test_preds_lr)), 
            {'fontsize': 10}, 
            fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('images/results/logistic_results.png')
    plt.clf()
    logging.info('LR report created and stored succesfully')
    
    return None

def feature_importance_plot(model, X_data, output_pth):
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
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title('Feature Importance')
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth + 'feature_importances.png')
    plt.clf()
    logging.info('Feature importance plot created and stored succesfully')


    return None

class ChurnModel:
    '''
    '''
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_text = None
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
            self.df = pd.read_csv(pth)

            self.df['Churn'] = self.df['Attrition_Flag']\
                .apply(lambda val: 0 if val == "Existing Customer" else 1)

            logging.info('Data imported succesfully')

            logging.info('The file has a shape of: {}'.format(self.df.shape))

            return self.df

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
        plot_hist(self.df, 'Churn', 'churn_distribution.png')
        logging.info('Churn distribution plot created and stored succesfully')
        plot_hist(self.df, 'Customer_Age', 'customer_age_distribution.png')
        logging.info('Customer Age distribution plot created and stored succesfully')
        plt.figure() 
        self.df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig('images/eda/marital_status_distribution.png')
        plt.clf()
        logging.info('Marital Status distribution plot created and stored succesfully')

        plt.figure() 
        sns.distplot(self.df['Total_Trans_Ct'])
        plt.savefig('images/eda/total_transaction_distribution.png')
        plt.clf()
        logging.info('Total Transaction distribution plot created and stored succesfully')

        plt.figure() 
        sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig('images/eda/heatmap.png')
        plt.clf()
        logging.info('Heatmap created and stored succesfully')

        return None

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
            cat_groups = self.df.groupby(i).mean()['Churn']

            for val in self.df[i]:
                cat_lst.append(cat_groups.loc[val]) 

            self.df[i+'_'+ response] = cat_lst

        logging.info('Encoder Helper executed succesfully')
        logging.info('Dataframe has a new shape of: {}'.format(self.df.shape))

        return self.df


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
        X = pd.DataFrame()
        X[constants.KEEP_COLS] = self.df[constants.KEEP_COLS]

        y = self.df['Churn']

        self.X_train, self.X_test, self.y_train, self.y_test =\
             train_test_split(X,
                            y,
                            test_size= 0.3, 
                            random_state=42)
        logging.info("Feature Engineering executed succesfully")
        logging.info("X_train has a shape of: {}".format(self.X_train.shape))
        logging.info("X_test has a shape of: {}".format(self.X_test.shape))
        logging.info("y_train has a shape of: {}".format(self.y_train.shape))
        logging.info("y_test has a shape of: {}".format(self.y_test.shape))

        return self.X_train, self.X_test, self.y_train, self.y_test

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

        param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }   

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(self.X_train, self.y_train)

        lrc.fit(self.X_train, self.y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.X_test)

        y_train_preds_lr = lrc.predict(self.X_train)
        y_test_preds_lr = lrc.predict(self.X_test)

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

        X_data = pd.concat([self.X_test, self.X_train], ignore_index=True)

        feature_importance_plot(rfc_model, X_data, './images/results/')

        # Plot and save ROC Curve
        lrc_plot = plot_roc_curve(lr_model, self.X_test, self.y_test)
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(rfc_model,
                                self.X_test, 
                                self.y_test, 
                                ax=ax, 
                                alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig('./images/results/roc_curve_result.png')
        plt.clf()
        logging.info('ROC curve plot has been succesfully created and stored')

        explainer = shap.TreeExplainer(rfc_model)
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test, plot_type="bar")
        plt.savefig('./images/results/shap_plot.png')
        plt.clf()
        logging.info('SHAP plot has been succesfully created and stored')

        return None

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


