import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the crime data
data = pd.read_csv('01_District_wise_crimes_committed_IPC_2001_2012.csv')
data.rename(columns={"KIDNAPPING & ABDUCTION":"K&A"},inplace=True)
data.rename(columns={"IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES":"IMG"},inplace=True)
data.rename(columns={"DOWRY DEATHS":"DD"},inplace=True)
# Keep only the selected columns
selected_columns = ['STATE/UT', 'DISTRICT', 'YEAR', 'MURDER', 'RAPE','K&A', 'ROBBERY', 'BURGLARY', 'THEFT', 'DD','IMG']
data = data[selected_columns]

# Drop rows with missing values
data.dropna(inplace=True)

data['sum_of_crime_numbers'] = data[selected_columns[3:]].sum(axis=1)

# Save the cleaned and prepared data to a new CSV file
data.to_csv('updated_cleaned_crime_data.csv', index=False)

# EXPLORATORY DATA ANALYSIS
# Confusion matrix correlation between numerical features
corr = data[['MURDER', 'RAPE', 'K&A', 'ROBBERY', 'BURGLARY','THEFT', 'DD', 'IMG']].corr()
f,axes = plt.subplots(1,1, figsize = (12,12))
sns.heatmap(corr , square= True, annot= True, linewidth= .9,center=2, ax=axes)
plt.setp(axes.xaxis.get_majorticklabels() , rotation = 45)
plt.setp(axes.yaxis.get_majorticklabels() , rotation = 45)
plt.show()

# Visualization for categorical data
print(data["STATE/UT"].value_counts())

# Histogram
fig=px.histogram(data, x="STATE/UT", width=1200, height=600, title="Visualizaton of State according to their Crime Type (Hover for the details)")
fig.show()

print(data["YEAR"].value_counts())

# Figure Size
fig, ax = plt.subplots(figsize=(7,7))

# Horizontal Bar Plot
title_cnt=data["YEAR"].value_counts().sort_values(ascending=False).reset_index()
mn = ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1],edgecolor='black', color=sns.color_palette('bright',len(title_cnt)))

# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=3)
ax.yaxis.set_tick_params(pad=3)

# Show top values
ax.invert_yaxis()

# Add Plot Title
ax.set_title('YEARLY TOTAL CRIME ANALYSIS 2001-2012',weight='bold',fontsize=20)
ax.set_xlabel('Count', weight='bold')

# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')
plt.xticks(weight='bold')
plt.show()

#Line graph
data = pd.read_csv('updated_cleaned_crime_data.csv')

# Select relevant columns for plotting
crime_types = ['MURDER', 'RAPE','K&A', 'ROBBERY', 'BURGLARY', 'THEFT', 'DD','IMG']

# Group the data by 'YEAR' and sum the values for each crime type
crime_data_grouped = data.groupby('YEAR')[crime_types].sum().reset_index()

# Plotting
plt.figure(figsize=(12, 6))

for crime_type in crime_types:
    plt.plot(crime_data_grouped['YEAR'], crime_data_grouped[crime_type], label=crime_type, marker='o')

plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.title('Trends of Incidents for Each Crime Type Over the Years')
plt.legend()
plt.grid(True)
plt.show()

dataset = pd.read_csv('updated_cleaned_crime_data.csv')

# Assuming 'sum_of_crime_numbers' is the target variable, and 'STATE/UT', 'DISTRICT', 'YEAR' are features
X = dataset[['STATE/UT', 'DISTRICT', 'YEAR']]
y = dataset[['MURDER', 'RAPE', 'K&A', 'ROBBERY', 'BURGLARY', 'THEFT', 'DD', 'IMG', 'sum_of_crime_numbers']]

# Identify categorical columns
categorical_cols = ['STATE/UT', 'DISTRICT']  # Add other categorical columns as needed

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Apply the column transformer to X
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Create separate decision tree regressors for each target variable
models = {}
for column in y.columns:
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train[column])
    models[column] = model

# Function to make predictions based on user input
def predict_crime(state, district, year):
    # Create a DataFrame with user input
    user_input = pd.DataFrame({'STATE/UT': [state], 'DISTRICT': [district], 'YEAR': [year]})

    # Apply the column transformer to user input
    user_input_preprocessed = preprocessor.transform(user_input)

    # Make predictions for each target variable
    predictions = {}
    for column, model in models.items():
        prediction = model.predict(user_input_preprocessed)
        predictions[column] = prediction[0]

    return predictions

# Example usage:
state_input = input("Enter the state: ")
district_input = input("Enter the district: ")
year_input = int(input("Enter the year: "))

# Preprocess user input
user_input_preprocessed = preprocessor.transform(pd.DataFrame({'STATE/UT': [state_input], 'DISTRICT': [district_input], 'YEAR': [year_input]}))

crime_predictions = predict_crime(state_input, district_input, year_input)
for crime_type, prediction in crime_predictions.items():
    print(f'{crime_type} number: {prediction}')

print('\n')
# Calculate MSE and R-squared for each target variable on the test set
for column, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test[column], y_pred)
    r2 = r2_score(y_test[column], y_pred)
    print(f'{column} - R-squared: {round(r2*100,2)}%')

