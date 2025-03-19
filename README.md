# Housing-Price-Prediction
# Housing Price Prediction using Machine Learning

## Overview
This project trains and evaluates machine learning models to predict housing prices using the Ames Housing dataset. The models include:
- A deep learning model using selected features
- A deep learning model using all features
- A multiple linear regression model

## Files Generated
Running the script will generate:
- `default_all_features.csv`: A default dataset row for reference
- `model_selected.h5`: Deep learning model trained on selected features
- `model_all.h5`: Deep learning model trained on all features
- `preprocessor_selected.pkl`: Preprocessing pipeline for selected features
- `preprocessor_all.pkl`: Preprocessing pipeline for all features

## Installation
To set up the environment, install the required dependencies:
```sh
pip install -r requirements.txt
```

## Running the Script
To generate the necessary files, run:
```sh
python run_pipeline.py
```

## Deploying on Streamlit Community Cloud
1. Upload the generated files and `app.py` to a GitHub repository.
2. Add `requirements.txt` to list dependencies.
3. Deploy on Streamlit Community Cloud by linking the `app.py` file in your GitHub repository.

## Notes
- Ensure the `AmesHousing.xlsx` dataset is available in the same directory as `run_pipeline.py` before running the script.

