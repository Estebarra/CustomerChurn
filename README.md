# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project seeks to predict the customer churn using features extracted from the credit card clients in a bank. The dataset used in this project is obtained from [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code). 

### Workflow
1- Import Data: Read the CSV File creating a Dataframe. \
2- Perform EDA: Generation of visualizations facilitating data analysis. \
3- Encoder Helper: Generation and integration of new columns useful to associate features with the target variable (Churn). \
4- Perform Feature Engineering: Creation of the training and testing dataset.\
5- Train Models: Training of multiple iterations of a Random Forest (Using a grid search) and a Logistic Regression using the previously mentioned features and the label data (Churn).

## Project structure:
```
.
├── data
    └── bank_data.csv
├── images
    ├── eda
        ├── churn_dustribution.png
        ├── customer_age_distribution.png
        ├── heatmap.png
        ├── marital_status_distribution.png
        └── total_transaction_distribution.png
    └── results
        ├── feature_importances.png
        ├── logistic_results.png
        ├── rf_results.png
        ├── roc_curve_result.png
        └── shap_plot.png
├── logs
    └── churn_library.log 
├── models
    ├── logistic_model.pkl
    └── rfc_model.pkl
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_test.py
├── constants.py
├── README.md
└── requirements.txt
```

### Folders description
- Data: Contains the csv file.
- Images: Is composed of two sub-folders (eda and results) containing the results of both the model training and the EDA. 
- Logs: Folder used to store the files where the logs generated by the project execution and testing are written.
- Models: Folder storing the ML models in a pickle format.

### Files description
- churn_library.py: Project's main module containing the workflow.
- churn_notebook.ipynb: Jupyter notebook where the workflow was created.
- churn_script_logging_and_tests.py: File holding a set of tests implemented using pytest.
- constants.py: As the name suggests, this file contains a set of constants used during the workflow execution.
- requirements.txt: File holding the dependencies required to execute the project.

## Running files

### Clonning the repository
```
git clone https://github.com/Estebarra/CustomerChurn.git
```

### Dependencies

[_requirements.txt_](requirements.txt)
```
shap==0.41.0
joblib==1.0.1
pandas==1.4.3
numpy==1.19.5
matplotlib==3.3.4
seaborn==0.11.2
sklearn==0.24.2
pytest==7.1.2
```

### Installing Dependencies
To install all the dependencies, execute the following command, just remember to update pip before installation.
```
pip install -r requirements.txt
```

### Running the workflow
To initialize and execute the workflow, input the following command while being in the main folder of the project
```
python3 churn_library.py
```

### Testing the code
To test the code and the workflow functions, use the following command. The testing must return 5 successes in order to confirm a correct workflow functionality.
```
pytest churn_script_logging_and_tests.py
```
