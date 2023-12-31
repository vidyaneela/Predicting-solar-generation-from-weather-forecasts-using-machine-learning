## PREDICTICTING SOLAR POWER GENERATION FROM WEATHER FORECAST USING MACHINE LEARNING
The need for more sustainable energy sources has grown as a result of the energy crisis.Increasing amount of households and companies are implementing solar panels to meet  their energy needs, making solar energy a key player in the market for alternative energy. Solar power sources are unpredictable by nature because the output power of PV systems is alternating and heavily dependent on environmental conditions. Planning ahead is essential for solar power generation due to the unpredictable nature of  photovoltaic systems.The objective of the solar power  project is to improve the efficiency and precision of solar power output prediction by utilizing machine learning models. weuse powerful ensemble models including Gradient Boosting and XGBoost regressors.Ensemble models are machine learning techniques that combine the predictions of multiple individual models to improve overall performance and robustness.

When utilizing Gradient Boosting Regressor and XGBoost for predicting solar generation from weather forecasts, selecting the right features is crucial for model performance. Here are some essential features that could be beneficial for these specific models:

1. **Weather Variables**:
      Solar radiation received by solar panels.
      Environmental temperature impacting panel efficiency.
      Temperature of the solar modules, influencing their performance.
      Water vapor in the air affecting panel efficiency.
     Influencing cooling effects on solar panels.

2. **Temporal Features**:
    Hourly, daily, or monthly time stamps to capture temporal patterns and seasonality.
   To identify weekly, monthly, or yearly trends.

3. **Derived Features**:
   Past values of solar generation and weather conditions.
    Moving averages, standard deviations over specific time windows.
   Grouping data into different time intervals to capture diurnal patterns.

4. **Interaction Features**:
   -Multiplicative or additive interactions between certain weather variables, considering how they affect each other's impact on solar generation.

5. **Geographical Features** (if available):
    Latitude, longitude impacting solar radiation levels.

6. **Additional Environmental Factors** (if available):
   Potential impact on solar panel efficiency.

This research included Solar power generation data and weather sensor data of a solar power plant
### Generation data:
  * DATE_TIME
  * PLANT_ID
  * DC_POWER
  * AC_POWER
  * DAILY_YIELD
  * TOTAL_YIELD
  ### weather sensor data:
  * DATE_TIME
  * PLANT_ID
  * AMBIENT_TEMPERATURE
  * MODULE_TEMPERATURE
    


## Requirements
->Python
->Environment with the specified Python libraries (Jupyter Notebooks, Google Colab)
->Required Python packages: pandas, pandas, matplotlib.

## Flow chart
![image](https://github.com/vidyaneela/Predicting-solar-generation-from-weather-forecasts-using-machine-learning/assets/94169318/68c9812d-398d-4262-b775-1317547431d1)


## Installation
-> Install required python libraries
->Install the required packages
```
pip install xgboost
```
->Install jupyter notebook for better usage


## Usage
->Open a new Google Colab notebook.

->Upload the project files in Google Drive.

->Load the pre-trained data sets. Ensure the model files are correctly placed in the Colab working directory.

->Execute the prediction of solar power generation script in the Colab notebook, which may involve adapting the script to run within a notebook environment.

->Follow the on-screen instructions or customize input cells in the notebook for the accuracy prediction.

->View and analyze the results directly within the Colab notebook.

## Program:

### IMPORTING AND LOADING DATA
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

```

### IMPORTING POWER GENERATION &WEATHER SENSOR DATA
```python
generation_data = pd.read_csv('/content/Plant_2_Generation_Data (1).csv')
weather_data = pd.read_csv('/content/Plant_2_Weather_Sensor_Data (1).csv')
generation_data.sample(5)
weather_data.sample(5)
```

### DATA PREPROCESSING
```python
generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'], format='%Y-%m-%d %H:%M')
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
```
### MERGING GENERATION DATA AND WEATHER SENSOR DATA
```python
df_solar = pd.merge(generation_data.drop(columns=['PLANT_ID']), weather_data.drop(columns=['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
df_solar.sample(5).style.background_gradient(cmap='cool')
```
### Adding separate time and date columns
```python
df_solar["DATE"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.date
df_solar["TIME"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.time
df_solar['DAY'] = pd.to_datetime(df_solar['DATE_TIME']).dt.day
df_solar['MONTH'] = pd.to_datetime(df_solar['DATE_TIME']).dt.month
df_solar['WEEK'] = pd.to_datetime(df_solar['DATE_TIME']).dt.week

df_solar['HOURS'] = pd.to_datetime(df_solar['TIME'], format='%H:%M:%S').dt.hour
df_solar['MINUTES'] = pd.to_datetime(df_solar['TIME'], format='%H:%M:%S').dt.minute
df_solar['TOTAL MINUTES PASS'] = df_solar['MINUTES'] + df_solar['HOURS']*60

df_solar["DATE_STRING"] = df_solar["DATE"].astype(str)
df_solar["HOURS"] = df_solar["HOURS"].astype(str)
df_solar["TIME"] = df_solar["TIME"].astype(str)

df_solar.head(2)

df_solar.info()


df_solar.isnull().sum()

df_solar.describe().style.background_gradient(cmap='rainbow')
```
### Converting 'SOURCE_KEY' from categorical form to numerical form
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_solar['SOURCE_KEY_NUMBER'] = encoder.fit_transform(df_solar['SOURCE_KEY'])
df_solar.head()
```
### Data Visualization:
```python
sns.displot(data=df_solar, x="AMBIENT_TEMPERATURE", kde=True, bins=100, color="red", facecolor="#3F7F7F", height=5, aspect=3.5);

df_solar['DATE'].nunique()
```
### Multiple Plotting of DC_POWER generation on per day basis:
```python
solar_dc = df_solar.pivot_table(values='DC_POWER', index='TIME', columns='DATE')

def Daywise_plot(data=None, row=None, col=None, title='DC Power'):
    cols = data.columns
    gp = plt.figure(figsize=(20,40))

    gp.subplots_adjust(wspace=0.2, hspace=0.5)
    for i in range(1, len(cols)+1):
        ax = gp.add_subplot(row,col, i)
        data[cols[i-1]].plot(ax=ax, color='red')
        ax.set_title('{} {}'.format(title, cols[i-1]), color='blue')
Daywise_plot(data=solar_dc, row=12, col=3)

daily_dc = df_solar.groupby('DATE')['DC_POWER'].agg('sum')

ax = daily_dc.sort_values(ascending=False).plot.bar(figsize=(17,5), legend=True, color='red')
plt.title('Daily DC Power')
plt.show()
```
###  Multiple Plotting of IRRADIATION generation on per day basis:
```python
solar_irradiation = df_solar.pivot_table(values='IRRADIATION', index='TIME', columns='DATE')

def Daywise_plot(data=None, row=None, col=None, title='IRRADIATION'):
    cols = data.columns
    gp = plt.figure(figsize=(20,40))

    gp.subplots_adjust(wspace=0.2, hspace=0.5)
    for i in range(1, len(cols)+1):
        ax = gp.add_subplot(row,col, i)
        data[cols[i-1]].plot(ax=ax, color='blue')
        ax.set_title('{} {}'.format(title, cols[i-1]), color='blue')
Daywise_plot(data=solar_irradiation, row=12, col=3)

daily_irradiation = df_solar.groupby('DATE')['IRRADIATION'].agg('sum')

daily_irradiation.sort_values(ascending=False).plot.bar(figsize=(17,5), legend=True, color='blue')
plt.title('IRRADIATION')
plt.show()
#In solar power plant DC_POWER or Output power is mostly depends on IRRADIATION .Or it is not wrong to say that it’s directly proportional.
```
###  Multiple Plotting of AMBIENT TEMPERATURE generation on per day basis:
```python
solar_ambient_temp = df_solar.pivot_table(values='AMBIENT_TEMPERATURE', index='TIME', columns='DATE')

def Daywise_plot(data=None, row=None, col=None, title='AMBIENT_TEMPERATURE'):
    cols = data.columns
    gp = plt.figure(figsize=(20,40))

    gp.subplots_adjust(wspace=0.2, hspace=0.5)
    for i in range(1, len(cols)+1):
        ax = gp.add_subplot(row,col, i)
        data[cols[i-1]].plot(ax=ax, color='darkgreen')
        ax.set_title('{} {}'.format(title, cols[i-1]), color='blue')

plt.figure(figsize=(16,16))
```
###  Highest average DC_POWER is generated on "2020-05-15"
```python
date=["2020-05-15"]

plt.subplot(411)
sns.lineplot(data=df_solar[df_solar["DATE_STRING"].isin(date)], x='DATE_TIME', y='DC_POWER', label="DC_Power_Best", color='green');

plt.title("DC Power Generation: {}" .format(date[0]))

plt.subplot(412)
sns.lineplot(data=df_solar[df_solar["DATE_STRING"].isin(date)], x='DATE_TIME', y='IRRADIATION', label="Irridation_Best", color='green');
plt.title("Irradiation : {}" .format(date[0]))

plt.subplot(413)
sns.lineplot(data=df_solar[df_solar["DATE_STRING"].isin(date)], x='DATE_TIME', y='AMBIENT_TEMPERATURE', label="Ambient_Temperature_Best", color='green');
sns.lineplot(data=df_solar[df_solar["DATE_STRING"].isin(date)], x='DATE_TIME', y='MODULE_TEMPERATURE', label="Module_Temperature_Best", color='blue');
plt.title("Module Temperature & Ambient Temperature: {}" .format(date[0]));

plt.tight_layout()
plt.show()
```
### Lowest average DC_POWER is generated on "2020-06-11"
```python
date=["2020-06-11"]
plt.figure(figsize=(16,16))

plt.subplot(411)
sns.lineplot(data=df_solar[df_solar["DATE_STRING"].isin(date)], x='DATE_TIME', y='DC_POWER', label="DC_Power_Worst", color='red');
plt.title("DC Power Generation: {}" .format(date[0]))

plt.subplot(412)
sns.lineplot(data=df_solar[df_solar["DATE_STRING"].isin(date)], x='DATE_TIME', y='IRRADIATION', label="Irridation_Worst", color='red');
plt.title("Irradiation : {}" .format(date[0]))

plt.subplot(413)
sns.lineplot(data=df_solar[df_solar["DATE_STRING"].isin(date)], x='DATE_TIME', y='AMBIENT_TEMPERATURE', label="Ambient_Temperature_Worst", color='red');
sns.lineplot(data=df_solar[df_solar["DATE_STRING"].isin(date)], x='DATE_TIME', y='MODULE_TEMPERATURE', label="Module_Temperature_Worst", color='blue');
plt.title("Module Temperature & Ambient Temperature: {}" .format(date[0]));

plt.tight_layout()
plt.show()
solar_dc_power = df_solar[df_solar['DC_POWER'] > 0]['DC_POWER'].values
solar_ac_power = df_solar[df_solar['AC_POWER'] > 0]['AC_POWER'].values

solar_plant_eff = (np.max(solar_ac_power) / np.max(solar_dc_power)) * 100
print(f"Power ratio AC/DC (Efficiency) of Solar Power Plant:  {solar_plant_eff:0.3f} %")
```
### Solar Power Plant Inverter Efficiency Calculation:
```python
#The inverter efficiency refers to how much dc power will be converted to ac power, as some of power will be lost during this transition in two forms:
Heat loss.
Stand-by power which consumed just to keep the inverter in power mode. Also, we can refer to it as inverter power consumption at no load condition.
Hence, inverter efficiency = pac/pdc where pac refers to ac output power in watt and pdc refers to dc input power in watts.

AC_list=[]
for i in df_solar['AC_POWER']:
    if i > 0:
        AC_list.append(i)

len(AC_list)

DC_list=[]
for i in df_solar['DC_POWER']:
    if i > 0:
        DC_list.append(i)

DC_list.sort()
DC_list.reverse()
len(DC_list)

plt.figure(figsize=(16,8))
AC_list.sort()
DC_list.sort()

eff = [i/j for i,j in zip(AC_list,DC_list)]

plt.plot(AC_list,eff,color='green')
plt.xlabel('Output power in kW')
plt.ylabel('efficiency AC/DC')
plt.title('Output power vs efficiency');
```
### Solar Power Prediction
```python
df2 = df_solar.copy()
X = df2[['DAILY_YIELD','TOTAL_YIELD','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','DC_POWER']]
y = df2['AC_POWER']

X.head()

y.head()
```
### Splitting the data
```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=21)
```
### MODEL CREATION
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
```
### Instantiate a Gradient Boosting Regressor
```python
gb_regressor = GradientBoostingRegressor()

# Fit the model to the training data
gb_regressor.fit(X_train, y_train)

# Predict the target variable on the testing data
y_pred_gb = gb_regressor.predict(X_test)

# Evaluate the model
r2_gb = r2_score(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
cross_val_scores_gb = cross_val_score(gb_regressor, X, y, cv=5)  # Cross-validation scores

# Print evaluation metrics
print(f'R2 Score (Gradient Boosting): {r2_gb:}')
print(f'Mean Squared Error (Gradient Boosting): {mse_gb:}')
print(f'Cross-Validation Scores (Gradient Boosting): {cross_val_scores_gb}')
```
### MODEL CREATION
```python
pip install xgboost

import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

# Instantiate an XGBoost Regressor
xgb_regressor = xgb.XGBRegressor()

# Fit the model to the training data
xgb_regressor.fit(X_train, y_train)

# Predict the target variable on the testing data
y_pred_xgb = xgb_regressor.predict(X_test)

# Evaluate the model
r2_xgb = r2_score(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
cross_val_scores_xgb = cross_val_score(xgb_regressor, X, y, cv=5)  # Cross-validation scores

# Print evaluation metrics
print(f'R2 Score (XGBoost): {r2_xgb:}')
print(f'Mean Squared Error (XGBoost): {mse_xgb:}')
print(f'Cross-Validation Scores (XGBoost): {cross_val_scores_xgb}')

prediction=xgb_regressor.predict(X_test)
print(prediction)

cross_checking = pd.DataFrame({'Actual' : y_test , 'Predicted' : prediction})
cross_checking.head()
```
### Visualize actual vs. predicted values
```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, alpha=0.5)
plt.title('Actual vs. Predicted Solar Power Generation')
plt.xlabel('Actual Solar Power Generation')
plt.ylabel('Predicted Solar Power Generation')
plt.show()
```
### Visualize actual vs. predicted values
```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.title('Actual vs. Predicted Solar Power Generation')
plt.xlabel('Actual Solar Power Generation')
plt.ylabel('Predicted Solar Power Generation')
plt.show()
```
### Output:
### Model evaluation metrics:
#### GradientBoostingRegressor:
![image](https://github.com/vidyaneela/Predicting-solar-generation-from-weather-forecasts-using-machine-learning/assets/94169318/5ed9b431-6d43-490f-879b-fc6620da7bb3)
#### XGBoost:
![image](https://github.com/vidyaneela/Predicting-solar-generation-from-weather-forecasts-using-machine-learning/assets/94169318/f85d4500-ccb8-48eb-a224-52ade8f7dbd4)

### Actual vs Predicted value:
![image](https://github.com/vidyaneela/Predicting-solar-generation-from-weather-forecasts-using-machine-learning/assets/94169318/1ef5a61c-3aea-4b73-8878-6b7b1903632d)

![image](https://github.com/vidyaneela/Predicting-solar-generation-from-weather-forecasts-using-machine-learning/assets/94169318/ccaafc3b-ea2a-448d-b6de-b2891cae4c27)


## Result:

These results suggest that thesolar power generation predicting model is both accurate and well-balanced, with high precision and recall values. Further analysis, including the examination of the confusion matrix and visualizations provide additional insights into the model's performance.



