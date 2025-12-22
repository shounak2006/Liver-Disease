# Binary Classification of Liver Disease using Machine Learning
## Project Overview
This project implements a machine learning–based decision support system for binary classification of liver disease using standard Liver Function Test (LFT) parameters.
The system is designed to assist in early risk screening by analyzing biochemical test values and providing a risk classification.

The application is deployed as a web-based interface using Streamlit, making it accessible from desktops and mobile devices via a browser.

Disclaimer: This system is intended strictly for screening assistance and academic purposes. It is not a diagnostic or clinical tool.

## Problem Statement
Liver diseases often remain undetected until advanced stages. Routine liver function tests generate multiple biochemical parameters, which can be challenging to interpret consistently.

This project aims to:

* Use machine learning to analyze LFT parameters

* Perform binary classification (High Risk / Low Risk)

* Provide a simple and interpretable decision-support interface

## Dataset 
### Indian Liver Patient Dataset (ILPD)
Source: Publicly available medical dataset
https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset

Records: 583

### Features:
* Age
  
* Gender

* Total Bilirubin

* Direct Bilirubin

* Alkaline Phosphatase

* Alamine Aminotransferase (ALT)

* Aspartate Aminotransferase (AST)

* Total Proteins

* Albumin

* Albumin and Globulin Ratio

### Target 

1 → Liver Disease

0 → No Liver Disease

## Methodology
### 1. Data Preprocessing
* Column renaming and correction

* Handling missing values using median imputation

* Encoding categorical variables (Gender)

* Binary target mapping
### 2. Model Training

#### Multiple models were evaluated:

* Logistic Regression

* Random Forest

* Support Vector Machine (SVM)

#### Logistic Regression was selected as the final model due to:

* Balanced performance

* High recall for liver disease cases

* Interpretability

#### 3. Evaluation Metrics

* Accuracy

* Precision

* Recall

* F1-score

* Confusion Matrix

## Application Interface (GUI)

The system includes a Streamlit-based graphical user interface that allows users to:

* Enter liver function test values

* View prediction results (High Risk / Low Risk)

* View risk probability

* Visualize entered test values using bar charts

The interface is responsive and can be accessed from mobile browsers.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8e15ff60-041d-4fa4-b058-82d61834faac" />
<img width="1000" height="1200" alt="image" src= "https://github.com/user-attachments/assets/d7d87860-a7d1-46e8-b8de-59095e918e1b" />

Analysed Data (Simulated)
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0115b0eb-a2f6-446b-969c-ad9b4a591e4d" />



## Deployment

The application is deployed publicly using Streamlit Community Cloud.

* Accessible via a web URL

* No local installation required for users

* Can be opened on desktop or mobile devices



# How to run the project on any PC
## 1. Prerequisites

* Python 3.9 or higher

* Git

## 2. Clone the Repository
git clone https://github.com/shounak2006/Liver-Disease.git  
cd Liver-Disease

## 3. (Recommended) Create Virtual Environment
python -m venv venv

### Activate it:

#### Windows

venv\Scripts\activate

#### macOS / Linux

source venv/bin/activate

## 4. Install Dependencies
pip install -r requirements.txt

## 5. Run the Application
streamlit run app.py

## Reproducibility

* All source code, trained model, dataset, and dependencies are included in the repository

* The project can be executed on any system by cloning the repo and installing dependencies

* The model can optionally be retrained using:

python src/train_model.py

## Results Summary

* Achieved approximately 72–74% accuracy

* High recall for liver disease cases

* Suitable for screening-oriented decision support

  
##Confusion Matrix
<img width="731" height="631" alt="image" src="https://github.com/user-attachments/assets/d6be9bd5-7aea-4af5-a2f0-7efb31ede482" />

Live Application URL: https://liver-disease-jamu8gabp6ftsbm5zpny9q.streamlit.app/




