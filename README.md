**Overview**
This repository contains the Python-based data science methodology and codebase for analyzing the cascading effects of financial toxicity in oncology. Using data from the Medical Expenditure Panel Survey (MEPS), this project models the statistical relationships between the financial burden of cancer care, the onset or exacerbation of depression, and subsequent treatment non-adherence.

The primary objective is to use biostatistical modeling to quantify how economic side effects impact mental health and patient compliance, identifying critical intervention points in patient care.

**Background & Motivation**
Cancer treatment often introduces severe economic burdens to patients, commonly referred to as "financial toxicity." This project explores the hypothesis that this financial strain acts as a catalyst for mental health decline (specifically depression), which in turn significantly increases the likelihood of a patient abandoning or missing their prescribed treatments. Understanding these links through robust data analysis is crucial for developing better holistic support systems for oncology patients.

**Dataset**
This analysis utilizes the Medical Expenditure Panel Survey (MEPS) dataset, a set of large-scale surveys of families and individuals, their medical providers, and employers across the United States.

**Key Variables Analyzed:**

Financial Burden: Out-of-pocket healthcare expenditures, income ratios, and insurance status.

Mental Health Indicators: Survey responses related to diagnosed depression or psychological distress markers.

Adherence Metrics: Self-reported missed treatments, inability to afford prescription medications, or delayed medical care.

Oncology Cohort: Patients with a reported cancer diagnosis.

(Note: The raw MEPS data files are not included in this repository due to size constraints. Instructions for downloading the data are provided below.)

Project Structure
Plaintext
meps-oncology-analysis/
│
├── data/                  # Directory for downloaded MEPS data (ignored in git)
├── notebooks/             # Jupyter notebooks for exploratory data analysis (EDA)
├── src/                   # Python scripts for data cleaning and modeling
│   ├── data_cleaning.py   # Scripts to preprocess MEPS survey data
│   └── modeling.py        # Statistical models and analysis scripts
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
Methodology
Data Preprocessing: Extracting the oncology cohort from the broader MEPS database, handling missing survey responses, and normalizing financial data.

Feature Engineering: Creating composite variables to represent "Financial Toxicity" (e.g., out-of-pocket costs relative to income) and "Non-Adherence."

Statistical Modeling: Utilizing Python libraries (Pandas, NumPy, statsmodels/scikit-learn) to perform regression analysis and identify significant correlations between the three primary pillars of the study.

Installation & Setup
To run this project locally, clone the repository and install the required dependencies:

Bash
# Clone the repository
git clone https://github.com/yourusername/meps-oncology-analysis.git

# Navigate into the directory
cd meps-oncology-analysis

# Install the required packages
pip install -r requirements.txt
Data Download:
Please download the relevant MEPS Full Year Consolidated Data Files from the AHRQ website and place them in the data/ directory before running the scripts.

Usage
To replicate the analysis, run the data cleaning script followed by the modeling script:

Bash
python src/data_cleaning.py
python src/modeling.py
Alternatively, you can step through the analysis interactively using the Jupyter notebooks located in the notebooks/ directory.

Future Work
Integrate longitudinal MEPS panel data to track financial toxicity over multiple years.

Expand the model to account for specific cancer subtypes and varied treatment regimens.

Apply machine learning techniques to predict patients at high risk of treatment non-adherence based on early financial indicators.

Author
Imanuel
BSc Cancer Biomedicine, University College London (UCL)
