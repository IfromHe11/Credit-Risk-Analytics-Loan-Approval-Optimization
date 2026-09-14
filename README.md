# 💳 Credit Risk Decision Engine

### Machine Learning · Credit Risk Analytics · Business Intelligence · Tableau

> **End-to-end machine learning project for predicting loan default risk, optimizing credit decisions, and measuring their financial impact on a lending portfolio.**

---

## 📌 Executive Summary

This project develops a **Credit Risk Decision Engine** for a hypothetical lending company.

The main business challenge is:

> **How can a lending company increase the number of approved loans while keeping credit losses within an acceptable risk level?**

Instead of treating credit risk as a traditional binary classification problem, this project connects the entire analytical process:

```text
Raw Data
   ↓
Data Quality
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Probability of Default
   ↓
Model Explainability
   ↓
Expected Loss
   ↓
Credit Decision
   ↓
Tableau Dashboard
```

The final solution is designed to support three possible decisions:

* 🟢 **APPROVE** — low predicted risk
* 🟡 **REVIEW** — medium predicted risk requiring additional assessment
* 🔴 **REJECT** — high predicted risk

The project focuses not only on predictive performance, but also on **business impact and risk-adjusted decision making**.

---

# 🎯 Business Problem

A lending company receives thousands of loan applications.

For every application, the company faces a trade-off:

### Approving a good customer

The company earns interest revenue and grows its loan portfolio.

### Approving a risky customer

The company may suffer a credit loss if the borrower defaults.

### Rejecting a good customer

The company avoids risk, but loses potential revenue and customer lifetime value.

Therefore, the objective is not simply:

> "Build the most accurate classification model."

The real objective is:

> **Find a credit decision policy that maximizes expected business value while maintaining an acceptable level of credit risk.**

---

# 🧠 Project Objectives

The project has six main objectives.

### 1. Understand the credit portfolio

Analyze:

* borrower characteristics;
* loan characteristics;
* credit history;
* loan purposes;
* risk grades;
* default patterns.

### 2. Identify key risk drivers

Determine which factors are associated with a higher probability of default.

### 3. Build predictive models

Train and compare several machine learning algorithms for predicting loan default.

### 4. Estimate Probability of Default

Convert model predictions into an interpretable:

**Probability of Default (PD)**

### 5. Connect ML with business economics

Estimate:

* Expected Loss;
* Expected Revenue;
* Expected Profit;
* Approval Rate;
* Portfolio Default Rate.

### 6. Build an interactive BI solution

Create Tableau dashboards that allow business users to:

* monitor portfolio risk;
* analyze customer segments;
* compare strategies;
* simulate different approval thresholds.

---

# 📊 Dataset

The project uses the **Credit Risk Dataset** published on Kaggle.

**Source:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data)

The dataset contains approximately:

* **32,581 loan applications**
* **12 original features**
* **1 target variable**

The target variable is:

```text
loan_status
```

where:

```text
0 → loan not defaulted
1 → loan defaulted
```

---

# 🗂️ Data Dictionary

| Feature                      | Description                  | Type        |
| ---------------------------- | ---------------------------- | ----------- |
| `person_age`                 | Applicant age                | Numerical   |
| `person_income`              | Applicant annual income      | Numerical   |
| `person_home_ownership`      | Home ownership status        | Categorical |
| `person_emp_length`          | Employment length in years   | Numerical   |
| `loan_intent`                | Purpose of the loan          | Categorical |
| `loan_grade`                 | Loan risk grade              | Categorical |
| `loan_amnt`                  | Loan amount                  | Numerical   |
| `loan_int_rate`              | Interest rate                | Numerical   |
| `loan_status`                | Default indicator            | Target      |
| `loan_percent_income`        | Loan amount as % of income   | Numerical   |
| `cb_person_default_on_file`  | Historical default indicator | Categorical |
| `cb_person_cred_hist_length` | Credit history length        | Numerical   |

---

# 🔎 Exploratory Data Analysis

The EDA phase focuses on understanding both the **customer population** and the **risk structure of the portfolio**.

## Data Quality

The analysis includes:

* missing values;
* duplicate observations;
* incorrect data types;
* unrealistic values;
* outliers;
* inconsistent categories;
* potentially problematic observations.

Example data-quality checks:

```python
df.isna().sum()
df.duplicated().sum()
df.describe()
df.nunique()
```

Special attention is paid to unrealistic applicant characteristics and extreme numerical values.

---

# 📈 Portfolio Analysis

The portfolio is analyzed from several perspectives.

### Applicant characteristics

* Age distribution
* Income distribution
* Employment length
* Home ownership

### Loan characteristics

* Loan amount
* Interest rate
* Loan purpose
* Loan grade
* Loan-to-income ratio

### Credit history

* Previous default
* Credit history length

---

# ⚠️ Default Risk Analysis

The key EDA question is:

> **Which customer and loan characteristics are associated with higher default rates?**

The analysis investigates default rates by:

* loan grade;
* loan purpose;
* income group;
* age group;
* loan amount;
* interest rate;
* loan-to-income ratio;
* home ownership;
* employment length;
* historical default status.

Example:

```text
Default Rate
    ↑
    │                         █
    │                    █    █
    │               █    █    █
    │          █    █    █    █
    │     █    █    █    █    █
    └────────────────────────────→
       A    B    C    D    E    F/G

              Loan Grade
```

The goal is to identify **risk concentration** rather than simply visualize distributions.

---

# 🧩 Risk Segmentation

Customers are segmented into risk groups using combinations of credit and loan characteristics.

Example conceptual segmentation:

| Segment        | Typical Characteristics                       | Risk   |
| -------------- | --------------------------------------------- | ------ |
| 🟢 Low Risk    | Low loan burden, strong credit profile        | Low    |
| 🟡 Medium Risk | Moderate debt burden and mixed credit profile | Medium |
| 🔴 High Risk   | High loan burden and adverse credit history   | High   |

The segmentation is used later for:

* portfolio analysis;
* business reporting;
* Tableau dashboards;
* decision strategy analysis.

---

# ⚙️ Feature Engineering

Several business-oriented features are derived from the original variables.

## Loan-to-Income Ratio

```python
loan_to_income = loan_amnt / person_income
```

Measures the size of the requested loan relative to annual income.

---

## Monthly Income

```python
monthly_income = person_income / 12
```

Used to create more interpretable affordability metrics.

---

## Employment-to-Age Ratio

```python
employment_ratio = person_emp_length / person_age
```

Provides additional context about employment stability.

---

## Credit History Ratio

```python
credit_history_ratio = cb_person_cred_hist_length / person_age
```

Measures the relative length of the applicant's credit history.

---

## Historical Default Flag

The historical default indicator is transformed into a machine-learning-friendly representation.

---

# 🤖 Machine Learning

The project compares multiple classification algorithms.

## Models

### Baseline

**Logistic Regression**

Chosen because it provides:

* strong interpretability;
* understandable coefficients;
* a useful benchmark;
* a traditional approach in credit risk modelling.

### Tree-Based Models

**Random Forest**

Used to capture:

* nonlinear relationships;
* feature interactions;
* complex decision boundaries.

### Gradient Boosting

**CatBoost / Gradient Boosting**

Used as an advanced model capable of capturing more complex patterns in the data.

---

# 🧪 Model Development Pipeline

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Train / Validation / Test
     │
     ▼
Preprocessing
     │
     ├───────────────┐
     ▼               ▼
Logistic        Tree-based
Regression       Models
     │               │
     └───────┬───────┘
             ▼
       Model Evaluation
             │
             ▼
        Calibration
             │
             ▼
    Probability of Default
```

---

📏 Model Evaluation

Accuracy is not considered the primary metric.

In credit risk, different types of prediction errors have different financial consequences.

Therefore, the following metrics are evaluated.

ROC-AUC

Measures the model's ability to distinguish between default and non-default cases.

PR-AUC

Particularly useful when evaluating the minority/default class.

Precision

Measures how many customers predicted as risky actually default.

Recall

Measures how many actual defaults are identified by the model.

F1 Score

Balances precision and recall.

Brier Score

Used to evaluate the quality of predicted probabilities.

Calibration

A credit-risk model should not only rank customers correctly.

Its probabilities should also be meaningful.

For example:

Predicted PD = 20%

should correspond approximately to a 20% observed default rate for a sufficiently large group of similar predictions.

🎯 Probability of Default

Instead of returning only:

Default = 0

or:

Default = 1

the final model produces:

Probability of Default (PD)

Example:

Applicant A → PD = 2.4%
Applicant B → PD = 8.7%
Applicant C → PD = 18.3%
Applicant D → PD = 42.1%

This probability becomes the main input for the business decision engine.

🔬 Model Explainability

A model used for credit decisions should provide understandable explanations.

The project uses SHAP to analyze model predictions.

The analysis answers:

Why did the model assign this applicant a high or low risk score?

Example:

Applicant PD = 31.7%

Factors increasing risk:
+ High loan-to-income ratio
+ Previous default
+ Higher interest rate
+ Riskier loan grade

Factors reducing risk:
- Higher income
- Longer credit history

SHAP is used for both:

Global Explainability

What are the most important risk drivers across the entire portfolio?

Local Explainability

Why was a specific applicant assigned a particular risk score?

💰 Business Metrics

The project translates ML predictions into financial metrics.

This is a key part of the project.

Probability of Default
PD = predicted probability of default
Expected Loss

A simplified credit-risk framework is used:

Expected Loss = PD × LGD × EAD

where:

PD  = Probability of Default
LGD = Loss Given Default
EAD = Exposure at Default

Because the dataset does not contain observed LGD and EAD values, explicit assumptions are used.

For example:

LGD = 60%
EAD = loan amount

Therefore:

expected_loss = pd * lgd * loan_amnt
💵 Expected Revenue

A simplified interest-revenue estimate is calculated using:

expected_revenue = loan_amnt * loan_int_rate

The exact business interpretation and assumptions are documented in the analysis notebook.

📈 Expected Profit

The project estimates:

Expected Profit
=
Expected Revenue
-
Expected Loss
-
Operating Costs

This allows the model to be evaluated from a business perspective rather than only from a statistical perspective.

🚦 Credit Decision Engine

The predicted Probability of Default is converted into an operational decision.

Conceptually:

                 Probability of Default
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Low Risk      Medium Risk    High Risk
             │             │             │
             ▼             ▼             ▼
         APPROVE         REVIEW        REJECT

The exact thresholds are not arbitrarily selected.

They are optimized using business metrics.

🎚️ Threshold Optimization

The model threshold determines how aggressively the company approves loans.

For every candidate threshold, the project calculates:

Approval Rate
Default Rate
Expected Loss
Expected Revenue
Expected Profit

The objective is:

MAXIMIZE Expected Profit

subject to:

Default Rate ≤ Risk Appetite

This transforms the ML model into a business decision system.

📊 Strategy Simulation

Three possible lending strategies are evaluated.

🛡️ Conservative Strategy

Focuses on minimizing credit risk.

Low PD threshold

Expected characteristics:

lower approval rate;
lower default rate;
lower expected losses;
potentially lower revenue.
⚖️ Balanced Strategy

Attempts to optimize the risk-return trade-off.

Expected characteristics:

moderate approval rate;
controlled default rate;
strong expected profitability.
🚀 Growth Strategy

Focuses on increasing loan approvals.

Expected characteristics:

higher approval rate;
higher revenue potential;
higher expected losses;
higher portfolio risk.
🏆 Business Recommendation

The final recommendation will be based on the strategy that provides the best balance between:

Profitability
      +
Portfolio Growth
      +
Risk Control

The selected strategy will be supported by quantitative evidence rather than model performance alone.

📊 Tableau Dashboard

The project includes an interactive Tableau dashboard designed for business users.

Dashboard 1 — Executive Overview

Designed for:

CEO;
Head of Risk;
Business Management.
KPIs
Total Applications
Approval Rate
Default Rate
Average Loan Amount
Expected Loss
Expected Revenue
Expected Profit
Visualizations
Portfolio overview
Default rate by loan grade
Default rate by loan purpose
Risk segment distribution
Expected Profit vs Risk
🔍 Dashboard 2 — Risk Analytics

Designed for Risk Managers and Analysts.

Filters
Loan Grade
Loan Intent
Home Ownership
Age Group
Income Group
Loan Amount
Historical Default
Visualizations
Default Rate by Loan Grade
Default Rate by Loan-to-Income
Default Rate by Interest Rate
Default Rate by Income
Risk Segment Analysis
Risk Driver Analysis
🎛️ Dashboard 3 — Decision Simulator

The dashboard includes a dynamic approval-threshold parameter.

Example:

PD Threshold

0% ─────────────●──────────── 30%
                 10%

Changing the threshold dynamically updates:

Approval Rate
Default Rate
Expected Loss
Expected Revenue
Expected Profit

This allows management to answer:

What happens to the portfolio if we become more conservative or more aggressive in lending?

💡 Business Value

The project demonstrates how machine learning can support real business decisions.

Potential business benefits include:

1. Better Risk Management

Identify high-risk applicants before loan approval.

2. Lower Expected Credit Losses

Avoid approving loans with unfavorable risk-return characteristics.

3. More Efficient Loan Approval

Automate the initial assessment of applications.

4. Improved Portfolio Management

Identify segments with concentrated credit risk.

5. Data-Driven Strategy Selection

Compare conservative, balanced, and growth lending strategies.

6. Explainable Decisions

Provide interpretable reasons behind risk predictions.

⚠️ Limitations

This project uses a public educational dataset and therefore does not represent a complete production-grade credit scoring system.

Important limitations include:

no actual loan repayment timeline;
no observed Loss Given Default;
no observed Exposure at Default;
no macroeconomic variables;
no transaction history;
no application timestamps;
no customer-level repayment behavior;
no real operational costs;
no real monetary loss observations.

Therefore, financial metrics such as Expected Loss and Expected Profit are scenario estimates based on explicit assumptions, rather than observed financial outcomes.

⚖️ Responsible Lending

Credit decisions can have significant consequences for individuals.

Therefore, a production system should also consider:

fairness;
explainability;
regulatory requirements;
model stability;
data drift;
monitoring;
human review;
protected characteristics;
adverse-action explanations.

The model developed in this project should be considered an analytical prototype, not a production credit approval system.

🛠️ Tech Stack
Programming
Python
SQL
Data Analysis
pandas
NumPy
SciPy
Visualization
Matplotlib
Seaborn
Plotly
Machine Learning
scikit-learn
CatBoost
Explainability
SHAP
BI
Tableau
Development
Jupyter Notebook
Git
GitHub
🔄 End-to-End Workflow
                    ┌─────────────────────┐
                    │    Credit Dataset   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Data Quality     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │         EDA         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │     Machine Learning     │
                  │                          │
                  │ Logistic Regression      │
                  │ Random Forest            │
                  │ CatBoost                 │
                  └────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation     │
                    │ ROC-AUC / PR-AUC     │
                    │ Recall / Precision   │
                    │ Calibration          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Probability of      │
                    │ Default (PD)        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      SHAP           │
                    │ Explainability      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Business Metrics    │
                    │                     │
                    │ Expected Loss       │
                    │ Revenue             │
                    │ Profit              │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Threshold           │
                    │ Optimization        │
                    └──────────┬──────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │       Decision Engine        │
                │                             │
                │ APPROVE │ REVIEW │ REJECT   │
                └──────────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Tableau        │
                    │     Dashboard      │
                    └─────────────────────┘
🚀 How to Run
1. Clone repository
git clone https://github.com/IfromHe11/Credit-Risk-Analytics-Loan-Approval-Optimization.git

2. Create virtual environment
python -m venv .venv
Windows
.venv\Scripts\activate
macOS / Linux
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Download the dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data

Place the CSV file into:

data/
5. Run notebooks

6. Tableau story:
https://public.tableau.com/app/profile/alexander.sergeevich/viz/Credit_Risk_analytics_loan_/Story1?publish=yes

📌 Key Questions Answered

The final project aims to answer the following business questions:

Risk

Who is most likely to default?

Drivers

What characteristics are associated with default risk?

Prediction

Can we reliably estimate Probability of Default?

Explainability

Why does the model consider an applicant risky?

Economics

How much money could the company expect to lose?

Strategy

Which approval threshold provides the best risk-return trade-off?

Management

How would changing the lending strategy affect the portfolio?

📈 Final Success Criteria

The project will be considered successful when it provides:

Analytical

Complete data quality analysis

Comprehensive EDA

Risk segmentation

Feature engineering

Identification of key risk drivers

Machine Learning

Baseline model

Multiple ML models

Cross-validation

Hyperparameter tuning

ROC-AUC

PR-AUC

Precision / Recall

Calibration

SHAP explainability

Business

Probability of Default

Expected Loss

Expected Revenue

Expected Profit

Threshold optimization

Scenario analysis

Recommended lending strategy

BI

Executive Dashboard

Risk Analytics Dashboard

Decision Simulator

Interactive filters

Business KPI monitoring

🏁 Final Outcome

The final deliverable is not simply a machine learning model.

It is an end-to-end Credit Risk Decision Engine that connects:

DATA
 ↓
ANALYTICS
 ↓
MACHINE LEARNING
 ↓
RISK
 ↓
FINANCIAL IMPACT
 ↓
BUSINESS DECISION
 ↓
VISUALIZATION

The core idea of the project is:

The best ML model is not necessarily the model with the highest ROC-AUC. The best model is the one that helps the business make better risk-adjusted decisions.

👨‍💻 Author

Alexander Sergeevich

Data Scientist / ML Engineer

Skills demonstrated
Python
Machine Learning
Credit Risk Analytics
Feature Engineering
Model Evaluation
Probability Calibration
SHAP
Business Analytics
Risk Management
Tableau
Data Visualization
Decision Science
⭐ Project Status
🟡 In Development

📚 Dataset Source

Credit Risk Dataset — Kaggle

https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data
