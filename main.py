# %% [markdown]
# # Assignment
# 
# ## Problem
# 
# The objective of this project is develop a predictive classifier to predict the next-day rain on the target variable RainTomorrow
# 
# ## Group Members
# 
# 
# | S.No. | Name | Student ID |
# | --- | --- | --- |
# |  1 | Hemang Sharma | 24695785 |
# |  2 |  Jyoti Khurana| 14075648 |
# |  3 | Mahjabeen Mohiuddin | 24610507 |
# |  4 | Suyash Santosh Tapase | 24678207 |
# 
# 
# ## Library used 
# 
# <ol>
#     <li>pandas</li>
#     <li>numpy</li>
#     <li>matplotlib</li>
#     <li>seaborn</li>
#     <li>plotly</li>
#     <li>sklearn</li>
# </ol>
# 
# 
# ### Link for DataSet & Source & Acknowledgements
# <ul><b>Observations were drawn from numerous weather stations</b>
#     <li>The daily observations are available from <a         href="http://www.bom.gov.au/climate/data">http://www.bom.gov.au/climate/data</a> </li>
#     <li>Definitions adapted from <a href="http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml">http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml</a> </li>
# </ul>
# <ul><b>Data source</b>
#     <li><a href="http://www.bom.gov.au/climate/data">http://www.bom.gov.au/climate/data</a></li>
#     <li><a href="https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data">https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data </a></li>
# </ul>

# %% [markdown]
# ## Importing packages
# We will import all the required packages and define our dataset

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# %% [markdown]
# ## Dataset
# In this step we will describe the data in the dataset
# 
# This dataset contains about 10 years of daily weather observations from many locations across Australia.
# 
# 
# ### Data Description
# <br>
# Location - Name of the city from Australia. .<br>
# MinTemp - The Minimum temperature during a particular day. (degree Celsius)<br>
# MaxTemp - The maximum temperature during a particular day. (degree Celsius)<br>
# MeanTemp - The mean temperature during a particular day. (degree Celsius)<br>
# Rainfall - Rainfall during a particular day. (millimeters)<br>
# Evaporation - Evaporation during a particular day. (millimeters)<br>
# Sunshine - Bright sunshine during a particular day. (hours)<br>
# WindGusDir - The direction of the strongest gust during a particular day. (16 compass points)<br>
# WindGuSpeed - Speed of strongest gust during a particular day. (kilometers per hour)<br>
# WindDir9am - The direction of the wind for 10 min prior to 9 am. (compass points)<br>
# WindDir3pm - The direction of the wind for 10 min prior to 3 pm. (compass points)<br>
# WindSpeed9am - Speed of the wind for 10 min prior to 9 am. (kilometers per hour)<br>
# WindSpeed3pm - Speed of the wind for 10 min prior to 3 pm. (kilometers per hour)<br>
# Humidity9am - The humidity of the wind at 9 am. (percent)<br>
# Humidity3pm - The humidity of the wind at 3 pm. (percent)<br>
# AvgHumidity - The average of humidity of the wind. (percent)<br>
# Pressure9am - Atmospheric pressure at 9 am. (hectopascals)<br>
# Pressure3pm - Atmospheric pressure at 3 pm. (hectopascals)<br>
# AvgPressure - The average Atmospheric pressure. (hectopascals)<br>
# Cloud9am - Cloud-obscured portions of the sky at 9 am. (eighths)<br>
# Cloud3pm - Cloud-obscured portions of the sky at 3 pm. (eighths)<br>
# Temp9am - The temperature at 9 am. (degree Celsius)<br>
# Temp3pm - The temperature at 3 pm. (degree Celsius)<br>
# RainToday - If today is rainy then ‘Yes’. If today is not rainy then ‘No’.
# RainTomorrow - This is will be the variable containing value of "if tomorrow is rainy then 1 (Yes) or if tomorrow is not rainy then 0 (No)"<br>

# %%
df_train = pd.read_csv('WeatherTrainingData.csv')
df_test = pd.read_csv('WeatherTestData.csv')

# %%
print(df_train.shape)
print(df_test.shape)

# %%
df_train

# %% [markdown]
# ## Data Cleaning
# 
# Now in order to use this data, we need to clean the data and remove all the empty cells from the dataset. So we will use dropna( )

# %%
data_test=df_test
data_train=df_train
data_test['RainToday'] = data_test['RainToday'].map({'Yes': 1, 'No': 0})

# %%
data_test

# %%
data_test.drop(columns=['Sunshine', 'Evaporation'], inplace=True)
categorical = data_test.select_dtypes(include = "object").columns
cleaner = ColumnTransformer([
    ('categorical_transformer', SimpleImputer(strategy='most_frequent'), categorical)
])
data_test[categorical] = cleaner.fit_transform(data_test[categorical])
null_columns=data_test.columns[data_test.isnull().any()]
data_test[null_columns].isnull().sum()

# %% [markdown]
# # Data Analysis
# 
# Now we will plot graphs comparing diffrent characteristics of our dataset 
# 
# ## 1. Feature Distribution

# %%
X = df_train.drop(columns=['RainTomorrow'])
y = df_train['RainTomorrow']
df = pd.concat([X, df_test], axis=0)

# %%
df.describe().T

# %%
df.drop(columns='row ID', inplace=True)
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data

# %%
df.drop(columns=['Sunshine', 'Evaporation'], inplace=True)
df.dtypes

# %%
categorical = df.select_dtypes(include = "object").columns
cleaner = ColumnTransformer([
    ('categorical_transformer', SimpleImputer(strategy='most_frequent'), categorical)
])
df[categorical] = cleaner.fit_transform(df[categorical])

null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()

# %%
df = df.fillna(df.median())
df.isnull().sum()

# %%
categorical = df.select_dtypes(include = "object").columns
for i in range(len(categorical)):
    print(df[categorical[i]].value_counts())
    print('************************************\n')

# %%
from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

df


# %%
'''objects = df.select_dtypes(include = "object").columns
for i in range(len(objects)):
    df[objects[i]] = LabelEncoder().fit_transform(df[objects[i]])

df'''

# %%
train = df.iloc[:99516,:]
new_train = pd.concat([train, y], axis=1)
test = df.iloc[99516:, :]
new_train

# %%
plt.figure(figsize=(17,18))
cor = new_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds,fmt='.2f')

# %%
sns.histplot(new_train['Humidity9am'])

# %%
sns.histplot(new_train['Humidity3pm'])

# %%
sns.boxplot(x=new_train['Cloud9am'])

# %%
sns.boxplot(x=new_train['Cloud3pm'])

# %%
sns.countplot(x=new_train['RainToday'])

# %%
new_train['RainTomorrow'].value_counts()

# %%
sns.countplot(x=new_train['RainTomorrow'])

# %%
df_majority_0 = new_train[(new_train['RainTomorrow']==0)] 
df_minority_1 = new_train[(new_train['RainTomorrow']==1)] 

df_minority_upsampled = resample(df_minority_1, 
                                 replace=True,    
                                 n_samples= 77157, 
                                 random_state=42) 

df_upsampled = pd.concat([df_minority_upsampled, df_majority_0])

# %%
plt.figure(figsize=(17,18))
cor = df_upsampled.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds,fmt='.2f')

# %%
sns.countplot(x=df_upsampled['RainTomorrow'])

# %%
sns.displot(data_test, x="MinTemp", hue='RainToday', kde=True)
plt.title("Minimum Temperature Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# The analysis revealed that the minimum temperature range from -8.5 ℃ to 33.9 ℃ and the minimum temperature of 11 ℃ had the highest frequency in the data set.

# %%
sns.displot(data_test, x="MaxTemp", hue='RainToday', kde=True)
plt.title("Maximum Temperature Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# On the other hand, the maximum temperature range from -4.1 ℃ to 48.1 ℃ and the maximum temperature of 20 ℃ has the highest frequency in the data set.

# %%
sns.displot(data_test, x="WindGustSpeed", hue='RainToday', kde=True)
plt.title("Wind Gust Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# During the analysis, it was found that the range of gusts was from 6 main points to 135 main points and 39.98 main points of gusts had the highest frequency in the data set.

# %%
sns.displot(data_test, x="Humidity9am", hue='RainToday', kde=True)
plt.title("Humidity at 9am Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# During the analysis, it was found that the range of air humidity at 9 o'clock in the morning. and at 3:00 p.m. from 0% to 100% and 99% humidity at 9:00 am. has the highest frequency in the data set.

# %%
sns.displot(data_test, x="Humidity3pm", hue='RainToday', kde=True)
plt.title("Humidity at 3pm Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# On the other hand, 54.43% of humidity at 3 pm has the highest frequency in the dataset.

# %%
sns.displot(data_test, x="Pressure9am", hue='RainToday', kde=True)
plt.title("Pressure at 9am Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# During the analysis, it was found that the range of wind pressure at 9 am. ranges from 980.5 hPa to 1042 hPa, and the pressure of 1017.68 hPa has the highest frequency in the data set.

# %%
sns.displot(data_test, x="Pressure3pm", hue='RainToday', kde=True)
plt.title("Pressure at 3pm Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# On the opposite hand, the variety of strain at three pm is from 978.2 hPa to 1039.6 hPa and 1015.28 hPa of strain has the very best frequency withinside the dataset.

# %%
sns.displot(data_test, x="Cloud9am", hue='RainToday', kde=True)
plt.title("Cloud at 9am Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# During the analysis, it's been determined that the variety of cloud at 9 am and 3 pm is from zero eighths to nine eighths and 4.44 eighths of cloud at nine am has the best frequency withinside the dataset.

# %%
sns.displot(data_test, x="Cloud3pm", hue='RainToday', kde=True)
plt.title("Cloud at 3pm Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# On the other hand, 4.52 eighths of cloud at 3 pm has the highest frequency in the dataset.

# %%
sns.displot(data_test, x="Temp9am", hue='RainToday', kde=True)
plt.title("Temperature at 9am Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# During the analysis, it has been found that the range of wind temperature at 9 am is from -7 ℃ to 40.2 ℃ and 17 ℃ of temperature has the highest frequency in the dataset.

# %%
sns.displot(data_test, x="Temp3pm", hue='RainToday', kde=True)
plt.title("Temperature at 3pm Distribution", fontsize = 14)
plt.show()

# %% [markdown]
# On the other hand, the range of pressure at 3 pm is from -5.1 ℃ to 46.7 ℃ and 27.68 ℃ of temperature has the highest frequency in the dataset.

# %%
df=data_test
sns.histplot(df['Humidity9am'])

# %%
sns.histplot(df['Humidity3pm'])

# %%
sns.histplot(df['Cloud9am'])

# %%
sns.histplot(df['Cloud3pm'])

# %%
sns.histplot(df['RainToday'])

# %%
x = list(data_test.MeanTemp)
y = list(data_test.Rainfall)
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(x, y, color ='blue',
        width = 0.4)
 
plt.xlabel("Average temperature (in degree Celsius)")
plt.ylabel("Rainfall during a particular day. (millimeters)")
plt.title("Relation between Average Temperature and Rainfall")
plt.show()

# %%
import seaborn as sns
import plotly.express as px

figure = px.scatter(data_frame = data_test, x="AvgHumidity",
                    y="MeanTemp", size="AvgHumidity", 
                    trendline="ols", 
                    labels={
                     "AvgHumidity": "Humidity (in percent)",
                     "MeanTemp": "Mean Temperature (in degree Celsius)"
                 },
                    title = "Relationship Between Temperature and Humidity")
figure.show()

# %% [markdown]
# ## 2. Average WindSpeed Analysis

# %%
windspeed_weather_df = data_test.groupby(['Location'])[['WindSpeed9am', 'WindSpeed3pm']].mean()
windspeed_weather_df = windspeed_weather_df.reset_index()
windspeed_weather_df.head()

# %%
x = windspeed_weather_df.loc[:, 'Location']
y1 = windspeed_weather_df['WindSpeed9am'] 
y2 = windspeed_weather_df['WindSpeed3pm']

plt.figure(figsize = (15, 8))

plt.plot(x, y1, marker='D', color = 'darkred', label = 'WindSpeed at 9am') 
plt.plot(x, y2, marker='D', color = 'blue', label = 'WindSpeed at 3pm')

plt.xlabel('Location', fontsize = 14)
plt.ylabel('WindSpeed', fontsize = 14)
plt.title('Location-wise observation of Average WindSpeed', fontsize = 18)
plt.legend(fontsize = 10, loc = 'best')
plt.xticks(rotation=80)
plt.show()

# %% [markdown]
# From this analysis, the wind speed at Melbourne Airport was determined to be the highest at 9:00 AM. with a speed of 20.29 km/h. On the other hand, at 3 o'clock in the afternoon. The highest wind speed is on the Gold Coast of Australia with 25.77 km/h. It can be concluded that the wind speed at 15:00. it is much higher than the wind speed at 9 o'clock in the morning.

# %% [markdown]
# ## 3. Average Humidity Analysis

# %%
humidity_weather_df = data_test.groupby(['Location'])[['Humidity9am', 'Humidity3pm']].mean()
humidity_weather_df = humidity_weather_df.reset_index()
humidity_weather_df.head()

# %%
x = humidity_weather_df.loc[:, 'Location']
y1 = humidity_weather_df['Humidity9am'] 
y2 = humidity_weather_df['Humidity3pm']

plt.figure(figsize = (15, 8))

plt.bar(x, y1, color = 'gold', label = 'Humidity at 9am') 
plt.bar(x, y2, color = 'blue',label = 'Humidity at 3pm')

plt.xlabel('Location', fontsize = 14)
plt.ylabel('Humidity', fontsize = 14)
plt.title('Location-wise observation of Average Humidity', fontsize = 18)
plt.legend(fontsize = 10, loc = 'best')
plt.xticks(rotation=80)
plt.show()

# %% [markdown]
# From this analysis it was found that the humidity of Dartmoor was highest at 9 am. 84.38%. On the other hand, at 3:00 p.m., Australia's Mount Ginnie has the highest humidity at 68.24%. In conclusion, it can be concluded that the humidity at 9 o'clock is much higher than the wind speed at 3 o'clock.

# %% [markdown]
# ## 4. Average Pressure Analysis

# %%
pressure_weather_df = data_test.groupby(['Location'])[['Pressure9am', 'Pressure3pm']].mean()
pressure_weather_df = pressure_weather_df.reset_index()
pressure_weather_df.head()

# %%
x = pressure_weather_df.loc[:, 'Location']
y1 = pressure_weather_df['Pressure9am'] 
y2 = pressure_weather_df['Pressure3pm']

plt.figure(figsize = (15, 8))

plt.plot(x, y1, marker='o', color = 'cyan', label = 'Pressure at 9am') 
plt.plot(x, y2, marker='o', color = 'darkcyan', label = 'Pressure at 3pm')

plt.xlabel('Location', fontsize = 14)
plt.ylabel('Pressure', fontsize = 14)
plt.title('Location-wise observation of Average Pressure', fontsize = 18)
plt.legend(fontsize = 10, loc = 'best')
plt.xticks(rotation=80)
plt.show()

# %% [markdown]
# During this analysis, it was found that the pressure in Canberra is the highest at 9 o'clock in the morning. at 1018.93 hPa. On the other hand, Adelaide, Australia has the highest pressure at 15:00 at 1016.79 hPa. In short, it can be concluded that the pressure at 9 o'clock is much higher than the wind speed at 3 o'clock.

# %% [markdown]
# ## 5. Average Temperature Analysis

# %%
location_weather_df = data_test.groupby(['Location'])[['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']].mean()
location_weather_df = location_weather_df.reset_index()
location_weather_df.head()

# %%
x = location_weather_df.loc[:, 'Location']
y1 = location_weather_df['MinTemp'] 
y2 = location_weather_df['MaxTemp']
y3 = location_weather_df['Temp9am'] 
y4 = location_weather_df['Temp3pm']

plt.figure(figsize = (15, 8))

plt.plot(x, y1, label = 'Minimum Temperature', marker='o', alpha = 0.8) 
plt.plot(x, y2, label = 'Maximum Temperature', marker='o', alpha = 0.8) 
plt.plot(x, y3, label = 'Temperature at 9am', marker='o', alpha = 0.8) 
plt.plot(x, y4, label = 'Temperature at 3pm', marker='o', alpha = 0.8)

plt.xlabel('Location', fontsize = 14)
plt.ylabel('Temperature', fontsize = 14)
plt.title('Location-wise observation of Average Temperature', fontsize = 18)
plt.legend(fontsize = 10, loc = 'best')
plt.xticks(rotation=80)
plt.show()

# %% [markdown]
# ## Models

# %%
X = df_upsampled.drop(columns='RainTomorrow')
y = df_upsampled['RainTomorrow']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True, random_state=44)

# %%
RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini', max_depth=17, n_estimators=100, random_state=44)
RandomForestClassifierModel.fit(X_train, y_train)


print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))
print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))

# %%
from sklearn.metrics import f1_score, accuracy_score

# Predict labels for training and test sets
y_train_pred = RandomForestClassifierModel.predict(X_train)
y_test_pred = RandomForestClassifierModel.predict(X_test)

# Calculate F1 score
f1 = f1_score(y_test, y_test_pred)
print('F1 Score:', f1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print('Accuracy:', accuracy)

# %%
import joblib
from sklearn.ensemble import RandomForestClassifier

# Build and train your model
# fit your model on training data

# Save your trained model
joblib.dump(RandomForestClassifierModel, 'RandomForestClassifierModel.joblib')


# %%
y_pred_RF = RandomForestClassifierModel.predict(X_test)
CM_RF = confusion_matrix(y_test, y_pred_RF)

sns.heatmap(CM_RF, center=True)
plt.show()

print('Confusion Matrix is\n', CM_RF)

# %%
GBCModel = GradientBoostingClassifier(n_estimators=200, max_depth=11, learning_rate=0.07, random_state=44)
GBCModel.fit(X_train, y_train)
print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

X = df_upsampled.drop(columns='RainTomorrow')
y = df_upsampled['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True, random_state=44)

GradientBoostingClassifierModel = GradientBoostingClassifier(n_estimators=200, max_depth=11, learning_rate=0.07, random_state=44)
GradientBoostingClassifierModel.fit(X_train, y_train)

# %%
print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))
# Predict labels for training and test sets
y_train_pred = GradientBoostingClassifierModel.predict(X_train)
y_test_pred = GradientBoostingClassifierModel.predict(X_test)

# Calculate F1 score
f1 = f1_score(y_test, y_test_pred)
print('F1 Score:', f1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print('Accuracy:', accuracy)


# %%
joblib.dump(GBCModel, 'GBCModel.joblib')

# %%
y_pred_GB = GBCModel.predict(X_test)
CM_GB = confusion_matrix(y_test, y_pred_GB)

sns.heatmap(CM_GB, center=True)
plt.show()

print('Confusion Matrix is\n', CM_GB)

# %%
y_pred = GBCModel.predict(test)

# %%
sns.countplot(y_pred)

# %%
test = pd.read_csv('WeatherTestData.csv')
submission = test[["row ID"]]
submission["RainTomorrow"] = y_pred

# %%
submission.to_csv('predict_weather.csv', index=False)

# %% [markdown]
# Two different testing algorithms that we use:
# 
# 1. Randomized Search Cross Validation for Hyperparameter Tuning:
# This algorithm randomly selects a set of hyperparameters and uses cross-validation to evaluate the model's performance. It then repeats this process multiple times and selects the best set of hyperparameters that give the highest accuracy score.

# %%
from sklearn.model_selection import RandomizedSearchCV

# define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# create a Random Forest Classifier object
rfc = RandomForestClassifier(random_state=42)

# create a RandomizedSearchCV object
rscv = RandomizedSearchCV(
    estimator=rfc, param_distributions=param_grid,
    n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1
)

# fit the RandomizedSearchCV object on the training data
rscv.fit(X_train, y_train)

# print the best hyperparameters and the corresponding accuracy score
print("Best Hyperparameters:", rscv.best_params_)
print("Best Accuracy Score:", rscv.best_score_)

# evaluate the model on the test data
rfc_best = rscv.best_estimator_
print("Test Accuracy Score:", rfc_best.score(X_test, y_test))

joblib.dump(rfc, 'rfc.joblib')

# %% [markdown]
# 2. Receiver Operating Characteristic (ROC) Curve:
# This algorithm is used to evaluate the performance of a binary classifier at different classification thresholds. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) for different threshold values. The area under the ROC curve (AUC-ROC) is a performance metric that ranges from 0.5 to 1. A higher AUC-ROC indicates better model performance.

# %%
from sklearn.metrics import roc_curve, auc

# fit the Gradient Boosting Classifier on the training data
gbc = GradientBoostingClassifier(n_estimators=200, max_depth=11, learning_rate=0.07, random_state=44)
gbc.fit(X_train, y_train)

# predict the probabilities of the positive class for the test data
y_proba = gbc.predict_proba(X_test)[:, 1]

# calculate the False Positive Rate (FPR), True Positive Rate (TPR), and threshold values
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# calculate the Area Under the Curve (AUC-ROC)
auc_roc = auc(fpr, tpr)

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc_roc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')

joblib.dump(gbc, 'gbc.joblib')

# %%
print(fpr, tpr)

# %% [markdown]
# The DummyClassifier in scikit-learn does not require explicit training or fitting since it employs simple rules for prediction based on the specified strategy. The strategy='most_frequent' strategy used in your code instructs the DummyClassifier to always predict the most frequent class in the training data. Hence, the model does not learn from the data during training.

# %% [markdown]
# DummyClassifier from scikit-learn, which provides a simple strategy for generating predictions.

# %%
# import necessary modules
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# create a new instance of the classifier
dummy = DummyClassifier(strategy='most_frequent')

# fit the model on the training data
dummy.fit(X_train, y_train)

# predict on the test data
y_pred_dummy = dummy.predict(X_test)

# evaluate the model
print('DummyClassifier Test Score is : ', dummy.score(X_test, y_test))

# calculate and print the confusion matrix
CM_dummy = confusion_matrix(y_test, y_pred_dummy)
sns.heatmap(CM_dummy, center=True)
plt.show()
print('Confusion Matrix is\n', CM_dummy)

# %%
'''import statistics
 
# list of positive integer numbers
MinTemp = data_test['MinTemp']
MaxTemp = data_test['MaxTemp']
Rainfall= data_test['Rainfall']
WindGustSpeed= data_test['WindGustSpeed']
WindSpeed9am= data_test['WindSpeed9am']
WindSpeed3pm= data_test['WindSpeed3pm']
Humidity9am= data_test['Humidity9am']
Humidity3pm= data_test['Humidity3pm']
Pressure9am= data_test['Pressure9am']
Pressure3pm= data_test['Pressure3pm']
Cloud9am =  data_test['Cloud9am']     
Cloud3pm = data_test['Cloud3pm']        
Temp9am = data_test['Temp9am']         
Temp3pm = data_test['Temp3pm']
RainToday = data_test['RainToday']

MinTemp_mean = statistics.mean(MinTemp)
MaxTemp_mean = statistics.mean(MaxTemp)
Rainfall_mean = statistics.mean(Rainfall)
WindGustSpeed_mean = statistics.mean(WindGustSpeed)
WindSpeed9am_mean = statistics.mean(WindSpeed9am)
WindSpeed3pm_mean = statistics.mean(WindSpeed3pm)
Humidity9am_mean = statistics.mean(Humidity9am)
Humidity3pm_mean = statistics.mean(Humidity3pm)
Pressure9am_mean = statistics.mean(Pressure9am)
Pressure3pm_mean = statistics.mean(Pressure3pm)
Cloud9am_mean  = statistics.mean(Cloud9am)      
Cloud3pm_mean  = statistics.mean(Cloud3pm)     
Temp9am_mean   = statistics.mean(Temp9am)      
Temp3pm_mean = statistics.mean(Temp3pm)

 
# Printing the mean
print("MinTemp Mean is :", MinTemp_mean)
print("MaxTemp Mean is :", MaxTemp_mean)
print("Rainfall Mean is :", Rainfall_mean)
print("WindGustSpeed Mean is :", WindGustSpeed_mean)
print("WindSpeed9am Mean is :", WindSpeed9am_mean)
print("WindSpeed3pm Mean is :", WindSpeed3pm_mean)
print("Humidity9am Mean is :", Humidity9am_mean)
print("Humidity3pm Mean is :", Humidity3pm_mean)
print("Pressure9am Mean is :", Pressure9am_mean)
print("Pressure3pm Mean is :", Pressure3pm_mean)
print("Cloud9am Mean is :", Cloud9am_mean)
print("Cloud3pm Mean is :", Cloud3pm_mean)
print("Cloud3pm Mean is :", Cloud3pm_mean)
print("Temp9am Mean is :", Temp9am_mean)
print("Temp3pm Mean is :", Temp3pm_mean)

MinTemp_standard_deviation = statistics.stdev(MinTemp)
MaxTemp_standard_deviation  = statistics.stdev(MaxTemp)
Rainfall_standard_deviation  = statistics.stdev(Rainfall)
WindGustSpeed_standard_deviation  = statistics.stdev(WindGustSpeed)
WindSpeed9am_standard_deviation  = statistics.stdev(WindSpeed9am)
WindSpeed3pm_standard_deviation  = statistics.stdev(WindSpeed3pm)
Humidity9am_standard_deviation  = statistics.stdev(Humidity9am)
Humidity3pm_standard_deviation = statistics.stdev(Humidity3pm)
Pressure9am_standard_deviation  = statistics.stdev(Pressure9am)
Pressure3pm_standard_deviation  = statistics.stdev(Pressure3pm)
Cloud9am_standard_deviation   = statistics.stdev(Cloud9am)      
Cloud3pm_standard_deviation  = statistics.stdev(Cloud3pm)     
Temp9am_standard_deviation   = statistics.stdev(Temp9am)      
Temp3pm_standard_deviation  = statistics.stdev(Temp3pm)


print("Standard Deviation of the MaxTemp is % s "%(statistics.stdev(MinTemp)))
print("Standard Deviation of the MaxTemp is % s "%(statistics.stdev(MaxTemp)))
print("Standard Deviation of the Rainfall is:", Rainfall_standard_deviation)
print("Standard Deviation of the WindGustSpeed is:",WindGustSpeed_standard_deviation)
print("Standard Deviation of the WindSpeed9am is:", WindSpeed9am_standard_deviation)
print("Standard Deviation of the WindSpeed3pm is:",WindSpeed3pm_standard_deviation)
print("Standard Deviation of the Humidity9am is:", Humidity9am_standard_deviation)
print("Standard Deviation of the Humidity3pm is:", Humidity3pm_standard_deviation)
print("Standard Deviation of the Pressure9am is:",Pressure9am_standard_deviation)
print("Standard Deviation of the Pressure3pm is:",Pressure3pm_standard_deviation)
print("Standard Deviation of the Cloud9am is:", Cloud9am_standard_deviation)
print("Standard Deviation of the Cloud3pm is:", Cloud3pm_standard_deviation)
print("Standard Deviation of the Temp9am is:", Temp9am_standard_deviation)
print("Standard Deviation of the Temp3pm is:",Temp3pm_standard_deviation)

import scipy.stats as stats

# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue= stats.f_oneway(data_test['MinTemp'], data_test['MaxTemp'],data_test['Rainfall'], data_test['WindGustSpeed'],data_test['WindSpeed9am'],data_test['WindSpeed3pm'],data_test['Humidity9am'],data_test['Humidity3pm'],data_test['Pressure9am'],data_test['Pressure3pm'],data_test['Cloud9am'],data_test['Cloud3pm'], data_test['Temp9am'],data_test['Temp3pm'])
pvalue = stats.f_oneway(data_test['MinTemp'], data_test['MaxTemp'],data_test['Rainfall'], data_test['WindGustSpeed'],data_test['WindSpeed9am'],data_test['WindSpeed3pm'],data_test['Humidity9am'],data_test['Humidity3pm'],data_test['Pressure9am'],data_test['Pressure3pm'],data_test['Cloud9am'],data_test['Cloud3pm'], data_test['Temp9am'],data_test['Temp3pm'])

#print(fvalue, pvalue)
print("The result of Anova test is:",fvalue)
print("The result of p vaue is:",pvalue)

#kruskal's test
result = stats.kruskal(data_test['MinTemp'], data_test['MaxTemp'],data_test['Rainfall'], data_test['WindGustSpeed'],data_test['WindSpeed9am'],data_test['WindSpeed3pm'],data_test['Humidity9am'],data_test['Humidity3pm'],data_test['Pressure9am'],data_test['Pressure3pm'],data_test['Cloud9am'],data_test['Cloud3pm'], data_test['Temp9am'],data_test['Temp3pm'])

# Print the result
print(result)


import statsmodels.api as sm
from statsmodels.formula.api import ols

#perform two-way ANOVA
model = ols('RainToday ~ MinTemp + MaxTemp + Rainfall + WindGustSpeed +WindSpeed9am +WindSpeed3pm +Humidity9am +Humidity3pm +Pressure9am +Pressure3pm +Cloud9am +Cloud3pm +Temp9am +Temp3pm', data=data_test).fit()
sm.stats.anova_lm(model, typ=2)

model = ols("""height ~ C(program) + C(gender) + C(division) +
               C(program):C(gender) + C(program):C(division) + C(gender):C(division) +
               C(program):C(gender):C(division)""", data=df).fit()

sm.stats.anova_lm(model, typ=2)'''


