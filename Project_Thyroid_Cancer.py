
"""
This script installs the pandas library, imports it, and loads thyroid cancer risk data from a CSV file into a 
pandas DataFrame. It then displays the first few rows of the DataFrame.
Functions:
    None
Variables:
    file_path (str): Path to the thyroid data CSV file.
    thyroid_data (DataFrame): DataFrame containing the thyroid cancer risk data.
Usage:
    Run the script to load the thyroid cancer risk data and display the first few rows.
"""
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns

# URL of the dataset
url = 'https://www.kaggle.com/datasets/ankushpanday1/thyroid-cancer-risk-prediction-dataset/download'

# Path to save the downloaded file
download_path = 'C:/Users/asadz/OneDrive/Documents/TMU-Project/thyroid_cancer_risk_data.csv'

# Download the dataset
#response = requests.get(url)
#with open(download_path, 'wb') as file:
#    file.write(response.content)

# Load the data into a pandas DataFrame
thyroid_data = pd.read_csv(download_path)

# Display the first few rows of the DataFrame
try:
    print(thyroid_data.head())
except pd.errors.ParserError as e:
    print(f"Error parsing the CSV file: {e}")

####### PREPROCESSING #######
# Display the column names of the DataFrame
print("Column names in the DataFrame:")
print(thyroid_data.columns)

#Display the shape of the DataFrame
print("Shape of the DataFrame:")
print(thyroid_data.shape)

#Display the data types of the columns
print("Data types of the columns:") 
print(thyroid_data.dtypes)

#Display the missing values in the DataFrame
missing_values = thyroid_data.isnull().sum()
print("Missing values in the DataFrame:", missing_values)

#Different countries in the dataset
countries = thyroid_data['Country'].unique()
print("Countries in the dataset:", countries)

def myPlot(data, var):
    plt.figure(figsize=(12, 8))
    sorted_data = thyroid_data.groupby([var, 'Diagnosis']).size().unstack().fillna(0)
    sorted_data['Total'] = sorted_data.sum(axis=1)
    sorted_data = sorted_data.sort_values(by='Total', ascending=False).drop(columns='Total')
    sorted_data.plot(kind='bar', stacked=True)
    plt.title('Thyroid Cancer Cases (Benign vs Malignant) by '+var)
    plt.xlabel(' ')
    plt.ylabel('Number of Cases')
    plt.xticks(rotation=18)
    plt.legend(title='Diagnosis')
    plt.show() 

myPlot(thyroid_data, 'Country')  
myPlot(thyroid_data, 'Ethnicity')
myPlot(thyroid_data, 'Gender')


# Proportion by categorical variables rounded to 2 decimal places
print("Proportion by categorical variables (rounded to 2 decimal places):")
categorical_columns = thyroid_data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"\n{column}:\n{thyroid_data[column].value_counts(normalize=True).round(2)*100}")  
    
    
# Summary statistics of all continuous variables rounded to 2 decimal places
continuous_columns = thyroid_data.select_dtypes(include=['number']).columns
continuous_columns_except_id = continuous_columns.drop('Patient_ID', errors='ignore')
summary_stats = thyroid_data[continuous_columns_except_id].describe().round(2)
print(summary_stats) 

    # Save the summary statistics to a LaTeX file
latex_file_path = 'C:/Users/asadz/OneDrive/Documents/TMU-Project/summary_statistics.tex'
with open(latex_file_path, 'w') as file:
    file.write(summary_stats.to_latex())

# Calculate the correlation matrix for continuous variables
correlation_matrix = thyroid_data[continuous_columns_except_id].corr()

# Display the correlation matrix
print("Correlation matrix of continuous variables:")
print(correlation_matrix)

# Save the correlation to a LaTeX file
latex_file_path = 'C:/Users/asadz/OneDrive/Documents/TMU-Project/correlation.tex'
with open(latex_file_path, 'w') as file:
    file.write(correlation_matrix.to_latex())

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Continuous Variables')
plt.show()

############## PCA ##############
import prince

# Perform PCA on the thyroid_data
dataset = prince.datasets.load_decathlon()
decastar=dataset.query('competition == "Decastar"')
print(decastar.head())
pca = prince.PCA(n_components=5)
pca = pca.fit(decastar, supplementary_columns=['rank', 'points'])
pca.eigenvalues_summary 

pca.transform(dataset).tail()
pca.plot(dataset)