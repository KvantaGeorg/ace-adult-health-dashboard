# ACE and Adult Health Dashboard

Interactive data analytics dashboard exploring the relationship between Adverse Childhood Experiences (ACEs) and adult health outcomes using BRFSS 2024 data.

## Project Overview

This project analyzes how childhood adversity is associated with adult mental health, physical health, substance use, healthcare access, protective factors, and depression risk.

The platform combines:
- Interactive data analytics
- Statistical analysis
- Machine learning
- AI-powered health insights
- Streamlit dashboard development

The dashboard was built to transform large-scale public health survey data into an accessible and interactive analytical experience.

## Dataset

Data source: Behavioral Risk Factor Surveillance System (BRFSS) 2024  
Official CDC dataset:
https://www.cdc.gov/brfss/annual_data/annual_2024.html

Dataset information:
- BRFSS 2024
- Total respondents: 457,670
- ACE-complete sample: 35,969 respondents

The full raw dataset is not included in this repository due to file size limitations.

## Dashboard Sections

- Overview
- Mental Health
- Physical Health
- Substance Use
- Healthcare Access
- Protective Factors
- Depression Risk
- AI Advisor

## Features

- Interactive Streamlit dashboard
- ACE prevalence analysis
- Mental health outcome analysis
- Physical health analytics
- Substance use analysis
- Healthcare access modeling
- Protective factor analysis
- Predictive depression risk estimation
- AI-powered health advisor
- Interactive visualizations and filtering
- Statistical testing and regression analysis

## Machine Learning

### Healthcare Access Prediction

A Logistic Regression model was trained to predict healthcare cost barriers using:
- Weighted ACE score
- Food insecurity
- Bill payment hardship
- Transportation barriers
- Income
- Age

The model was evaluated using:
- ROC-AUC
- Confusion matrix
- Classification metrics

### Depression Risk Prediction

A Neural Network model was trained to estimate depression risk using:
- ACE score
- Age group
- Sex
- Income level
- Loneliness indicators

The model generates personalized risk estimates based on BRFSS population-level patterns.

## AI Advisor

The project includes an AI-powered ACE Risk Advisor built using:
- Groq API
- Llama 3.3 70B Versatile

The AI Advisor:
- Explains ACE-related health risks
- Interprets statistical findings
- Provides personalized insight summaries
- Responds conversationally to user questions

The chatbot uses real statistics generated from the BRFSS analysis pipeline.

## Key Insights

- Higher ACE exposure is associated with significantly worse mental health outcomes.
- Depression prevalence increases substantially across ACE groups.
- Respondents with higher ACE scores report more poor physical health days.
- Higher ACE scores are associated with increased substance use indicators.
- Healthcare access barriers rise with ACE exposure and socioeconomic hardship.
- Emotional support and protective adult relationships reduce some negative outcomes.
- Loneliness strongly predicts depression outcomes.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Statsmodels
- SciPy
- Joblib
- Groq API
- Llama 3.3

## My Contribution

This project was developed as part of a collaborative research initiative.

My primary responsibilities included:

- Designing and developing the full Streamlit dashboard
- Building the dashboard structure, layout, and interactive user experience
- Developing the complete Mental Health analytics section
- Selecting and implementing visualizations for the sections I developed
- Creating dashboard logic and interactive filtering systems
- Training and integrating predictive machine learning models
- Developing the Depression Risk prediction feature
- Building and integrating the AI Advisor chatbot
- Implementing personalized user interaction flows

The broader statistical analysis and several analytical sections were developed collaboratively within the team.

## Limitations

- BRFSS data is self-reported
- ACEs may be underreported
- ACE weighting methodology was subjective
- Correlation does not imply causation
- Binary ACE variables do not capture duration or severity
- The ACE sample represents only participating states
- Some potentially important variables were unavailable

## Future Improvements

- Add subgroup analysis by demographic groups
- Improve predictive model performance
- Add explainable AI functionality
- Deploy the dashboard publicly
- Improve preprocessing automation
- Add additional health outcome models
- Expand longitudinal analysis

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
