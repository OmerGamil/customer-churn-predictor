# Customer Churn Predictor - A Predictive Classification Model for Determining Customer Churn

[Customer Churn Predictor](https://customer-churn-predictor.herokuapp.com/) is a machine-learning (ML) project using the IBM Telco Customer Churn dataset to determine whether an ML pipeline could be built to predict whether a customer is at risk of leaving a telecommunications company. This was achieved using a classification task, using the Churn attribute from the dataset as the target and the remaining customer attributes as features.

## Table of Contents

- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis](#hypothesis-and-how-to-validate)
- [Mapping Business Requirements to Data Visualisation and ML Tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Epics and User Stories](#epics-and-user-stories)
- [Dashboard Design](#dashboard-design)
- [Technologies Used](#technologies-used)
- [Testing](#testing)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
- [Credits](#credits)
- [Acknowledgements](#acknowledgements)


## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). Each row represents a customer and each column contains a customer attribute. The dataset includes information about:
    - customer demographics such as gender, age group, and family status
    - account information including contract type, payment method, and billing
    - services subscribed to, including phone, internet, and add-on services
    - whether or not the customer churned (left the company)

| Attribute | Information | Units |
|---|---|---|
| customerID | Unique customer identifier | String |
| gender | Customer gender | Male, Female |
| SeniorCitizen | Whether customer is a senior | 0: No, 1: Yes |
| Partner | Whether customer has a partner | Yes, No |
| Dependents | Whether customer has dependents | Yes, No |
| tenure | Months with the company | Integer (0-72) |
| PhoneService | Phone service subscription | Yes, No |
| MultipleLines | Multiple phone lines | Yes, No, No phone service |
| InternetService | Internet service type | DSL, Fibre optic, No |
| OnlineSecurity | Online security add-on | Yes, No, No internet service |
| OnlineBackup | Online backup add-on | Yes, No, No internet service |
| DeviceProtection | Device protection add-on | Yes, No, No internet service |
| TechSupport | Technical support add-on | Yes, No, No internet service |
| StreamingTV | TV streaming service | Yes, No, No internet service |
| StreamingMovies | Movie streaming service | Yes, No, No internet service |
| Contract | Contract term type | Month-to-month, One year, Two year |
| PaperlessBilling | Paperless billing status | Yes, No |
| PaymentMethod | Payment method | Electronic check, Mailed check, Bank transfer, Credit card |
| MonthlyCharges | Monthly charge amount | Numeric (USD) |
| TotalCharges | Total amount charged | Numeric (USD) |
| Churn | Target: whether customer left | Yes: churned, No: retained |

[Back to top](#table-of-contents)

## Business Requirements

* IBM estimates that acquiring a new customer costs five to seven times more than retaining an existing one. A fictional telecommunications company has requested a data practitioner to analyse a dataset of customer accounts in order to determine which attributes are most associated with churn and whether customer data can accurately predict whether a customer will leave.

* Business Requirement 1 - The client is interested in which attributes correlate most closely with customer churn, i.e. what are the most common risk factors?
* Business Requirement 2 - The client is interested in using customer data to predict whether or not a customer will churn.

[Back to top](#table-of-contents)

## Hypothesis and how to validate?

* Hypothesis 1:
    - We suspect that customers on month-to-month contracts are significantly more likely to churn than those on one-year or two-year contracts.
    - **Validation**: A bar chart of churn rate per contract type will indicate a strong relationship between contract length and the target Churn.

* Hypothesis 2:
    - We suspect that customers with higher monthly charges are more likely to churn.
    - **Validation**: A Pearson correlation analysis between MonthlyCharges and the binary Churn column, supported by a distribution plot comparing churners vs non-churners.

* Hypothesis 3:
    - We suspect that customers with lower tenure (fewer months with the company) are at the highest risk of churning.
    - **Validation**: A Pearson correlation analysis between tenure and the binary Churn column, supported by a box plot comparing tenure distributions for churners vs non-churners.

[Back to top](#table-of-contents)

## The rationale to map the business requirements to the Data Visualizations and ML tasks

* **Business Requirement 1**: Data Visualisation and Correlation study
    - We need to perform a correlation study to determine which features correlate most closely with the target.
    - A Pearson correlation heatmap will indicate linear relationships between numerical variables and the Churn target.
    - Distribution plots (box plots and histograms) will be created for the top correlated numeric features: tenure and MonthlyCharges.
    - Bar charts of churn rate will be created for key categorical features: Contract type, TechSupport, and InternetService.
    - This will be carried out during the **Data Visualization, Cleaning, and Preparation** Epic (see Epics & User Stories).

* **Business Requirement 2**: Classification Model
    - We need to predict whether a customer is at risk of churning or not.
    - Therefore we need to build a binary classification model.
    - A conventional machine learning pipeline will be able to map the relationships between the features and target.
    - Extensive hyperparameter optimisation will give us the best chance at a highly accurate prediction.
    - A decision threshold will be tuned on the validation set to ensure the model achieves at least 75% recall on the Churn class.
    - This will be carried out during the **Model Training, Optimization and Validation** Epic (see Epics & User Stories).

[Back to top](#table-of-contents)

## ML Business Case

**Classification Model**

* We want a ML model to predict whether a customer will churn based on previously gathered customer data. The target variable, 'Churn', is categorical and contains two classes: 0 (no churn) and 1 (churn).
* We will consider a **classification model**, a supervised model with a two-class, single-label output that matches the target.
* The model success metrics are:
    - at least 75% recall for churn on the train and test sets
* The model will be considered a failure if:
    - the model fails to achieve 75% recall for the Churn class
* The model output is defined as a flag, indicating if a customer will churn or not and the associated probability of churn.
* The training data to fit the model comes from: [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
    - The dataset contains: 7,043 observations and 21 attributes.
    - Target: Churn; Features: all other attributes.

[Back to top](#table-of-contents)

## Epics and User Stories

* The project was split into 5 Epics based upon the Data Visualisation and Machine Learning tasks and within each of these, user stories were set out to enable an agile methodology.

### Epic - Information Gathering and Data Collection

* **User Story** - As a data analyst, I can import the dataset from Kaggle so that I can save the data in a local directory.
* **User Story** - As a data analyst, I can load and inspect the saved dataset so that I can understand its structure and data quality before any processing begins.

### Epic - Data Visualization, Cleaning, and Preparation

* **User Story** - As a data practitioner, I can clean the raw dataset so that all downstream notebooks and the ML model receive consistent, properly typed data.
* **User Story** - As a data scientist, I can visualise the dataset so that I can interpret which attributes correlate most closely with customer churn (**Business Requirement 1**).
* **User Story** - As a data analyst, I can evaluate which transformations best prepare the features for modelling so that the ML pipeline applies the right encoding and scaling to each column type.

### Epic - Model Training, Optimization and Validation

* **User Story** - As a data scientist, I can split the data into train, validation, and test sets to prepare it for the ML model.
* **User Story** - As a data engineer, I can train a GradientBoosting classification pipeline with SMOTE oversampling so that the model learns to identify churners despite the class imbalance (**Business Requirement 2**).
* **User Story** - As a data engineer, I can carry out an extensive hyperparameter optimisation to ensure the ML model gives the best results (**Business Requirement 2**).
* **User Story** - As a data engineer, I can tune the decision threshold on the validation set so that the model achieves at least 75% recall on the Churn class (**Business Requirement 2**).

### Epic - Dashboard Planning, Designing, and Development

* **User Story** - As a business stakeholder, I can view a project summary that describes the project, dataset and business requirements to understand the project at a glance.
* **User Story** - As a business stakeholder, I can view the project hypotheses and validations to determine what the project was trying to achieve and whether it was successful.
* **User Story** - As a retention team member, I can enter a customer's details and get an instant churn prediction so that I can take proactive action before the customer leaves (**Business Requirement 2**).
* **User Story** - As a technical user, I can view the correlation analysis to see how the outcomes were reached (**Business Requirement 1**).
* **User Story** - As a technical user, I can view all the data to understand the model performance and see statistics related to the model (**Business Requirement 2**).

### Epic - Dashboard Deployment and Release

* **User Story** - As a user, I can view the project dashboard on a live deployed website.
* **User Story** - As a technical user, I can follow instructions in the readme to fork the repository and deploy the project for myself.

[Back to top](#table-of-contents)

## Dashboard Design

### Page 1: Project Summary

* **Section 1 - Summary**
    * Introduction to the project
    * Description of the dataset, where it was sourced
    * Link to the readme
* **Section 2 - Business Requirements**
    * Description of both business requirements

### Page 2: EDA and Correlation Study

* State business requirement 1
* Overview of dataset - display churn distribution and describe dataset shape
* Display Pearson correlation heatmap for numeric features
* Display distributions of correlated features against target (tenure, MonthlyCharges)
* Display bar charts for key categorical features (Contract, TechSupport, InternetService)
* Conclusions

### Page 3: Project Hypotheses

* Outline the three project hypotheses
* Present validation and outcome for each hypothesis

### Page 4: Model Performance

* Summary of the ML pipeline architecture
* Optimal decision threshold value
* Precision-recall trade-off chart
* Confusion matrix for the test set
* Classification report table
* Feature importance chart
* Business requirement outcome statement

### Page 5: Churn Predictor

* State business requirement 2
* Widget inputs for all 19 customer attributes
* "Predict" button to run inputted data through the ML model and output a prediction and probability

[Back to top](#table-of-contents)

## Technologies Used

The technologies used throughout the development are listed below:

### Languages

* [Python](https://www.python.org/)

### Python Packages

* [Pandas](https://pandas.pydata.org/docs/index.html) - Open source library for data manipulation and analysis.
* [Numpy](https://numpy.org/doc/stable/index.html) - Adds support for large, multi-dimensional arrays and matrices, and high-level mathematical functions.
* [Matplotlib](https://matplotlib.org/) - Comprehensive library for creating static, animated and interactive visualisations.
* [Seaborn](https://seaborn.pydata.org/) - Data visualisation library for drawing attractive and informative statistical graphics.
* [Plotly](https://plotly.com/python/) - Interactive charting library.
* [Feature-engine](https://feature-engine.trainindata.com/en/latest/) - Library with multiple transformers to engineer and select features for machine learning models.
* [scikit-learn](https://scikit-learn.org/stable/) - Open source machine learning library that features various algorithms for training a ML model.
* [SciPy](https://scipy.org/) - Library used for scientific computing and technical computing.
* [Imbalanced-learn](https://imbalanced-learn.org/stable/) - Provides tools for dealing with classification problems with imbalanced classes.
* [Joblib](https://joblib.readthedocs.io/en/stable/) - Provides tools for lightweight pipelining, e.g. caching output values.
* [Streamlit](https://streamlit.io/) - Framework for building and sharing data applications.
* [Pillow](https://pillow.readthedocs.io/en/stable/) - Python imaging library for loading plots onto the dashboard.

### Other Technologies

* [Git](https://git-scm.com/) - For version control
* [GitHub](https://github.com/) - Code repository and GitHub projects was used as a Kanban board for Agile development
* [Heroku](https://heroku.com) - For application deployment
* [Gitpod](https://gitpod.io/) - IDE used for development
* [Kaggle](https://kaggle.com/) - Source of the training dataset

[Back to top](#table-of-contents)

## Testing

### Manual Testing

#### User Story Testing

* Dashboard was manually tested using user stories as a basis for determining success.
* Jupyter notebooks were reliant on consecutive functions being successful so manual testing against user stories was deemed irrelevant.

*As a business stakeholder, I can view a project summary that describes the project, dataset and business requirements to understand the project at a glance.*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Project summary page | Viewing summary page | Page is displayed with dataset description and both business requirements | Functions as intended |

---

*As a business stakeholder, I can view the project hypotheses and validations to determine what the project was trying to achieve and whether it was successful.*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Project hypotheses page | Navigate to page | Clicking on navbar link in sidebar navigates to correct page | Functions as intended |
| Hypothesis content | View page | All three hypotheses displayed with validation evidence and outcome | Functions as intended |

---

*As a retention team member, I can enter a customer's details and get an instant churn prediction (Business Requirement 2).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Churn predictor page | Navigate to page | Clicking on navbar link in sidebar navigates to correct page | Functions as intended |
| Enter live data | Interact with widgets | All 19 input widgets are interactive and respond to user input | Functions as intended |
| Live prediction | Click on "Predict" button | Clicking button displays customer profile and churn prediction result | Functions as intended |

---

*As a technical user, I can view the correlation analysis to see how the outcomes were reached (Business Requirement 1).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| EDA and Correlation Study page | Navigate to page | Clicking on navbar link in sidebar navigates to correct page | Functions as intended |
| Churn distribution | View page | Churn distribution bar chart displayed with interpretation | Functions as intended |
| Correlation heatmap | View page | Pearson correlation heatmap displayed | Functions as intended |
| Feature distributions | View page | Distribution plots for tenure and MonthlyCharges displayed | Functions as intended |
| Categorical charts | View page | Bar charts for Contract, TechSupport, InternetService displayed | Functions as intended |

---

*As a technical user, I can view all the data to understand the model performance and see statistics related to the model (Business Requirement 2).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Model performance page | Navigate to page | Clicking on navbar link in sidebar navigates to correct page | Functions as intended |
| Pipeline overview | View page | Pipeline architecture and optimal threshold displayed | Functions as intended |
| Precision-Recall chart | View page | Threshold tuning chart with 75% target line displayed | Functions as intended |
| Confusion matrix | View page | Confusion matrix for the test set displayed | Functions as intended |
| Classification report | View page | Precision, recall, and F1-score table displayed | Functions as intended |
| Feature importance | View page | Feature importance bar chart displayed | Functions as intended |
| Business requirement outcome | View page | Statement confirming whether 75% recall target was met | Functions as intended |

---

### Validation

All code in the app_pages and src directories was validated as conforming to PEP8 standards using CodeInstitute's PEP8 Linter.

| File | Result |
| --- | --- |
| `app.py` | No errors |
| `app_pages/multipage.py` | No errors |
| `app_pages/page_summary.py` | No errors |
| `app_pages/page_eda.py` | No errors |
| `app_pages/page_hypothesis.py` | No errors |
| `app_pages/page_model_performance.py` | No errors |
| `app_pages/page_churn_predictor.py` | No errors |
| `src/data_management.py` | No errors |
| `src/machine_learning/evaluate_clf.py` | No errors |
| `src/machine_learning/predictive_analysis.py` | No errors |

### Automated Unit Tests

No automated unit tests have been carried out at this time.

[Back to top](#table-of-contents)

## Unfixed Bugs

* At the time of writing, there are no unfixed bugs within the project.

[Back to top](#table-of-contents)

## Deployment

### Heroku

* The App live link is: [Customer Churn Predictor](https://customer-churn-predictor.herokuapp.com/)

The project was deployed to Heroku using the following steps:

1. Within your working directory, ensure there is a setup.sh file containing the following:
```
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```
2. Within your working directory, ensure there is a runtime.txt file containing a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack supported version of Python.
```
python-3.12.1
```
3. Within your working directory, ensure there is a Procfile file containing the following:
```
web: sh setup.sh && streamlit run app.py
```
4. Ensure your requirements.txt file contains all the packages necessary to run the streamlit dashboard.
5. Update your .gitignore and .slugignore files with any files/directories that you do not want uploading to GitHub or are unnecessary for deployment.
6. Log in to [Heroku](https://id.heroku.com/login) or create an account if you do not already have one.
7. Click the **New** button on the dashboard and from the dropdown menu select "Create new app".
8. Enter a suitable app name and select your region, then click the **Create app** button.
9. Once the app has been created, navigate to the Deploy tab.
10. At the Deploy tab, in the Deployment method section select **GitHub**.
11. Enter your repository name and click **Search**. Once it is found, click **Connect**.
12. Navigate to the bottom of the Deploy page to the Manual deploy section and select main from the branch dropdown menu.
13. Click the **Deploy Branch** button to begin deployment.
14. The deployment process should happen smoothly if all deployment files are fully functional. Click the button **Open App** at the top of the page to access your App.
15. If the build fails, check the build log carefully to troubleshoot what went wrong.

[Back to top](#table-of-contents)

## Forking and Cloning

If you wish to fork or clone this repository, please follow the instructions below:

### Forking

1. In the top right of the main repository page, click the **Fork** button.
2. Under **Owner**, select the desired owner from the dropdown menu.
3. **OPTIONAL:** Change the default name of the repository in order to distinguish it.
4. **OPTIONAL:** In the **Description** field, enter a description for the forked repository.
5. Ensure the 'Copy the main branch only' checkbox is selected.
6. Click the **Create fork** button.

### Cloning

1. On the main repository page, click the **Code** button.
2. Copy the HTTPS URL from the resulting dropdown menu.
3. In your IDE terminal, navigate to the directory you want the cloned repository to be created.
4. In your IDE terminal, type ```git clone``` and paste the copied URL.
5. Hit Enter to create the cloned repository.

### Installing Requirements

In order to ensure all the correct dependencies are installed in your local environment, run the following command in the terminal:

    pip install -r requirements.txt

[Back to top](#table-of-contents)

## Credits

### Content

#### Data Collection Notebook
* The dataset was sourced from the IBM Telco Customer Churn dataset, available on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

#### Exploratory Data Analysis Notebook
* The code for the correlation heatmap and distribution plots was inspired by the Code Institute "Churnometer" walkthrough project.

#### Feature Engineering Notebook
* The custom function for analysing feature transformations was adapted from the Code Institute "Data Analytics Packages - ML: feature-engine" module.

#### Modelling And Evaluation Notebook
* The custom functions for hyperparameter optimisation and displaying the confusion matrix were adapted from the Code Institute "Data Analytics Packages - ML: Scikit-learn" module.
* Decision threshold tuning using precision_recall_curve was implemented following guidance from the scikit-learn documentation.

#### Streamlit Dashboard
* The multi-page class was taken from the Code Institute "Data Analysis & Machine Learning Toolkit" streamlit lessons.

[Back to top](#table-of-contents)

## Acknowledgements

* Thanks to my mentor Mo Shami, for his support and guidance on the execution of the project

[Back to top](#table-of-contents)

---
