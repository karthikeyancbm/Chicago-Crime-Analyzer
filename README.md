# Chicago-Crime-Analyzer

# Problem Statement:

The primary objective of this project is to leverage historical and recent crime data to identify patterns, trends, and hotspots within Chicago.The task is to provide actionable insights that can shape our crime prevention strategies, ensuring a safer and more secure community.

# Approach:

**1.Data Wrangling:**

It involves the following steps:

  * Data Cleaning
  * Data Transforamtion
  * Data Enrichment

**Data Cleaning:**

  * Handle missing values with mean/median/mode.
  * Treat Outliers using IQR
  * Identify Skewness in the dataset and treat skewness with appropriate data transformations,
    such as log transformation.
  * Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding,
    or ordinal encoding, based on their nature and relationship with the target variable.
**Data transformation:**

  *  Changing the structure or format of data, which can include normalizing data, scaling features, or encoding categorical variables.

**Exploaratory Data Analysis:**

  * Analyse the past data and to identify crime patterns, trends and hotspots and explore it through visualization vide different plots
    using matplotlib and folium map.
  * Recommendations have been given according to the insights observed.

**Feature Engineering:**

  * Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data.

**Model Building and Evaluation:**

* Split the dataset into training and testing/validation sets.
* Train the different regression models and evaluate the result with suitable metrics such as MAE - Mean Absolute Error, MSE - Mean Squared Error and
  RMSE - Root Mean Squared Error.

**Model Deployment using Streamlit:**
* Develop interactive GUI using streamlit.
* create an input field where the user can enter each column value except target variable.

    
