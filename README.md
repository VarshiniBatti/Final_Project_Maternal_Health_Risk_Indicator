# Final Project Report 
### For the html files, please use the predictor.html and visualization.html files in the main branch instead of in templates
## Introduction
The topic that was chosen to explore in the final project is maternal health. According to WHO, in 2020, almost 800 women died each day from completely preventable causes that are related to pregnancy and childbirth(World Health Organization, n.d.). Specifically, in the same year, approximately 95% of all maternal deaths occurred in low and middle income countries(World Health Organization, n.d.). The UN lists the topic of maternal mortality as the third goal in the Sustainable Development Goals (SDG) and it aims to reduce global mortality rates to less than 70 per 100,000 live births by 2030(UN Women, n.d.). While these goals have been stated, little progress has been made in terms of actually reducing mortality rates. Maternal health and mortality is a very significant matter, however, it is seldom discussed when talking about existent issues in the world, even though it is preventable. Therefore, in order to shed light on this issue, the aim of the project was to develop a machine learning model that predicts the intensity of health risk for patients that are going through pregnancy using several biological and physiological variables. 

The dataset that has been used for the project is from Kaggle, and has originally been collected by researchers investigating health risk factors for pregnant women in rural areas of Bangladesh through an IoT based maternal healthcare system, with sensors for heart rate, blood pressure, blood glucose levels, and other variables. The dataset was collected from the IoT device, local hospitals, and web portals in Bangladesh, and stored in a csv file. For this project, the csv file was acquired using a file download. The data set was interesting because it had recorded distinct physiological features that are commonly measured to predict the risk level of the mother. 

The data provider, Kaggle, was in compliance with the FAIR principles. The dataset was findable for many reasons. Since there were different physiological variables present, such as blood sugar concentration, blood pressure, body temperature, and heart rate, the data provider clearly specified the metadata and the units used for each factor clearly on the page. The dataset did not contain unique identifiers for patients, and just listed out their values without any identifying personal information. The dataset is easily searchable through Kaggle. The dataset is also accessible because in Kaggle, the dataset is under the Attribution 4.0 International license by Creative Commons, which allows for sharing and adapting the dataset to derive insights with open access as long as the appropriate credit is given to the authors. The dataset is from the UC Irvine Machine Learning Repository, and is from a research paper investigating maternal health risk factors. The dataset is in the form of a CSV file. The dataset is interoperable because it uses the same units for all of its factors. Most of the factors are represented in SI units, for example blood sugar concentration is in mmol/L, and heart rate is in beats per minute. The Creative Commons license is clear in that it states that it allows for the dataset to be shared, copied, transformed, and adapted, therefore it is also reusable. The dataset was well-annotated with metadata and the licensing terms were clear. 
## Data Exploration 
The dataset contained 1014 rows and 7 columns, which are age in years, systolic and diastolic blood pressure in mmHg, body temperature in Fahrenheit, blood sugar concentration(represented as BS) in mmol/L, and heart rate in beats per minute. The dataset did not have NA values. Therefore, Then, in order to look at the distribution and the range of data, histograms were plotted for each column, whose images are shown below: 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_1.png" alt="" width="400" height="400">

With these plots, it is observed that majority of the women in the dataset were between the ages of 15 and 35 years, most of the blood sugar concentration levels were in between 6 and 8 mmol/L, the heart rates were mostly between 62 to 80 bpm, and most of the women had a body temperature of 98 degrees F. The systolic and diastolic blood pressure graphs are more spread out, with varied values. According to the CDC, the normal values for the systolic blood pressure are around 120 mmHg and for the diastolic pressure are around 80 mmHg. The patient is considered to have elevated blood pressure of hypertension if their systolic blood pressure is greater than 130 mmHg and if their diastolic blood pressure is greater than 80 mmHg(Center of Disease Prevention and Control, n.d.). To delve deeper, the dataset was then filtered by risk level to look at the distribution of data by risk level. The age distribution per risk level can be observed in the graphs below. 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_2.png" alt="" width="400" height="400">
<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_3.png" alt="" width="400" height="400">
<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_4.png" alt="" width="400" height="400">

The histograms show that most of the low risk women fall between 10 to 30 years of age, most of the mid risk women fall between 20 to 30 years of age, and most of the high risk women fall under 30 to 50 years of age. These results follow the domain knowledge that as a woman’s age increases, the health risk of pregnancy also increases as well. In the dataset, there are 406 women who have been classified as low risk, 336 women as mid risk, and 272 women as high risk. The summary statistics for each of the physiological features are provided in the image below. 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_12.png" alt="" width="400" height="400">

The summary statistics show that the average of women in the dataset is 29.8, the average systolic and diastolic blood pressure are 113.2 and 76.4 respectively, the average blood sugar concentration is around 8.72 mmol/L, the average body temperature is 98.6 degrees F, and the average heart rate is about 74 beats per minute. Among these features, age, systolic BP and diastolic BP have the most spread, since they have higher standard deviation values (13.5, 18.4, and 13.8 respectively). 
To check the spread of the data when it comes to ages in different risk level categories, box plots were constructed for each level, which is shown in the image below. 
<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_5.png" alt="" width="400" height="400">

The spread for each level shows that women at medium and high risk are women who also fall between 30 and 50 years of age. One interesting observation was that there are outliers in the ages, specifically, there are plenty of women that are in their 60s and 70s and are still at low risk. There are some women who are medium risk who are in their 50s and 60s as well. These outliers skew the data for the age of women in the low risk to the right. 
Research shows that increased blood sugar concentration of the mother during pregnancy can negatively affect the health of the baby. Extreme cases of hyperglycemia(high blood glucose), can lead to gestational diabetes which can affect the baby’s health and even the timing of the baby’s birth(Zhao et. al, 2023). Therefore, box plots were also made for the blood sugar concentrations based on risk level, which are shown in the image below.

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_6.png" alt="" width="400" height="400">

The blood sugar concentration levels for women at low risk and medium risk had outliers in the data, and the outliers mostly are higher blood sugar concentration levels. These outliers skewed the data for these categories to the right. Boxplots were also done for systolic and diastolic blood pressures, as shown in the images below. 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_11.png" alt="" width="400" height="400">

Both the boxplots follow roughly the same trend: as risk increases, both the systolic and the diastolic blood pressure values also increase. 
## Data Analysis and Results 
The first data analysis that was run was a correlation matrix between the physiological features of the dataset. The results are shown in the image below. 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_7.png" alt="" width="400" height="400">

In the matrix, each variable’s correlation to another variable is based on the scale from -1 to 1, with -1 indicating a strong negative correlation and +1 indicating a strong positive correlation. For age, the strongest and most positive correlation is with blood sugar concentration, which has a correlation of 0.47 and vice versa. The strongest negative correlation for age is with body temperature, with a correlation of -0.26. For systolic blood pressure, the strongest correlation, at 0.79, is with diastolic blood pressure and vice versa. The strongest negative correlation for systolic blood pressure and diastolic blood pressure is with body temperature, with a correlation of -0.29 and -0.26 respectively. For body temperature the strongest positive correlation is with heart rate, at 0.10. 

### Machine Learning Models 
For the project, the machine learning models that were used to run on the dataset were Random Forests and XGBoost. Both machine learning techniques are tree-based models and were chosen because of their ability to give results with high accuracy and their compatibility with large datasets. The main objective of the project was to develop a tool that could predict the mother’s health risk in pregnancy using a variety of physiological variables. The results would output the risk level, which can either be high risk, mid or medium risk, or low risk. 
#### Random Forest Model 
The first machine learning model that was run on the cleaned dataset was the random forest model. The target variable, which is risk level, was first label encoded into numerical values of 0,1, and 2 which indicate low risk, mid risk, and high risk respectively. Then the features were separated from Risk Level and the data was split into training and testing sets of 80-20 ratio. The number of trees to be generated was chosen to be 100, and the place where to start randomly generating the trees would be 42. Randomizing the starting point allows for the model to be reused and still achieve the same results. The model was trained and feature importances were extracted, and the feature with the highest importance was blood sugar concentration, with a feature importance score of 0.369. The model was tested and the accuracy of the model came out to be 0.83. Since the number of women for each risk level is imbalanced, a better predictor of accuracy would be F1 scores. Classification reports were used to validate the analyses in the Random Forest model. The classification report shows the F1 score for high risk is 0.86, for low risk is 0.83, and for mid risk is 0.81. This means that the model was able to mostly accurately predict the true positive cases in each risk level, while minimizing the false positives. The feature importance graph developed by the random forest model is shown below. 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_8.png" alt="" width="400" height="400">

The confusion matrix shows that 40 cases were correctly identified as low risk, 64 cases were correctly identified as mid risk, and 64 cases were correctly classified as high risk. The total cases for the low risk level were 47, for the mid risk level were 80, and high risk level were 76. The model predicted 46 out of the 47 cases to be low risk, 74 out of 80 cases to be mid risk, and 83 out of 76 cases as high risk. Both the classification report and the confusion matrix for the Random Forest model are shown below. 

Classification Report for Random Forest Model 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_13.png" alt="" width="400" height="400">

Confusion Matrix for Random Forest Model 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_9.png" alt="" width="400" height="400">


#### XGBoost Model
The second machine learning model that was run on the cleaned dataset was the XGBoost model. The target variable, risk level, was label encoded into numerical values in the same manner as with the Random Forest model. Then the features were separated from Risk Level and the data was split into training and testing sets of 80-20 ratio. An XGBoost classifier is run with multiclass softmax, since there are multiple risk levels in the dataset. The model was then trained and tested, and the accuracy of the model came out to be 0.84. Classification reports were used to validate the analyses in the XGBoost model as well.  In the report, the F1 scores for high risk were 0.86, for low risk were 0.85, and for mid risk were 0.83. This means that the XGBoost model was slightly better at classifying the correct levels of risk for patients and minimized a large amount of false positives as well. In the confusion matrix, the model predicted 41 cases of high risk, 66 cases of low risk, and 64 cases of mid risk correctly. The model predicted 48 out of the 47 cases to be low risk, 76 out of the 80 cases to be mid risk, and 79 out of the 76 cases to be high risk. Overall, the values from the F1 scores and from the confusion matrix indicate that the XGBoost model was a more accurate predictor of maternal health risk level compared to Random Forest. Therefore, the XGBoost model was the final model to be deployed for the website of the predictor tool. Both the classification report and the confusion matrix for the XGBoost model are shown below. 

Classification Report for XGBoost Model 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_14.png" alt="" width="400" height="400">

Confusion Matrix for XGBoost Model 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_10.png" alt="" width="400" height="400">

## Server API and Web Interface 
The goal of the website was to introduce the predictive model in a user-friendly manner, while providing more information about the features, and the general issue about maternal health and maternal mortality. The website has four pages: the homepage, the predictor tool, the visualizations, and a contact page. The homepage consists of general statistics and information about maternal health, a link to the Github repository for all the files of the project, and a navigation bar that shows all the webpages. The predictor model is another webpage on the website, and it takes in the age, systolic and diastolic blood pressure, heart rate, and blood sugar concentration of the patient and returns the risk level of the patient as a result page. The prediction also has specific boundaries for the values of each input and if the values are outside those ranges, then it returns an error message in the form of an error page. Another webpage on the website is the data visualizations page, which has all the visualizations that have been done in the project. Lastly, to make the website mimic a real-world website user design interface, there is also a contact page for users as well. 
## Challenges  
One of the unexpected difficulties encountered in the project was finding a dataset that would fit the criteria required to run machine learning models to develop a predicting tool. Most of the datasets on maternal health had licensing issues or other issues with the data and the size, therefore the dataset explored in the project was the closest one to fitting the requirements of having an ample amount of data for machine learning models to run on and predict with accuracy. There were a limited number of features in the dataset, which might not be ideal when holistically assessing a patient and coming up with ideal treatment options. Therefore, a further area of research would be to see the performance of the model if there are more features added, especially socioeconomic factors and factors about social identity such as race and ethnicity, because these factors do have an effect on the health of the mother and the baby. Socioeconomic factors do play a significant role especially in access to care, so it would be interesting to see the effects of these factors on the health risk of patients. Another challenge is that the geographic location of the women in the dataset is very specific, in that it only considers women in rural areas of Bangladesh. It would be more interesting to see if the predictor model used data from other locations all over the world or even just for comparing women who live in rural and urban areas to see the differences in health risk. Another challenge would be that there were also a relatively smaller amount of data points, therefore increasing the sample size would improve the F1 scores and the accuracy of the ML models in predicting the risk level.  
## Acknowledgements
The dataset used for the project is from Kaggle and is originally from the University of California Irvine’s Machine Learning Repository. The dataset is under the Attribution 4.0 International license, which permits open access and the sharing, adapting, and transforming of the dataset. The data has originally been collected from researchers looking into using IoT sensors as a basis for a maternal healthcare system that could predict the risk level of the mother based on physiological variables in rural areas of Bangladesh. 
The citation of the paper is: M. Ahmed and M. A. Kashem, "IoT Based Risk Level Prediction Model For Maternal Health Care In The Context Of Bangladesh," 2020 2nd International Conference on Sustainable Technologies for Industry 4.0 (STI), Dhaka, Bangladesh, 2020, pp. 1-6, doi: 10.1109/STI50764.2020.9350320.
## References
Centers for Disease Control and Prevention. (2021, May 18). High blood pressure symptoms and causes. Centers for Disease Control and Prevention. https://www.cdc.gov/bloodpressure/about.htm

M. Ahmed and M. A. Kashem, "IoT Based Risk Level Prediction Model For Maternal Health Care In The Context Of Bangladesh," 2020 2nd International Conference on Sustainable Technologies for Industry 4.0 (STI), Dhaka, Bangladesh, 2020, pp. 1-6, doi: 10.1109/STI50764.2020.9350320.

SDG 3: Ensure healthy lives and promote well-being for all at all ages. UN Women – Headquarters. (n.d.). https://www.unwomen.org/en/news/in-focus/women-and-the-sdgs/sdg-3-good-health-well-being#:~:text=By%202030%2C%20reduce%20the%20global,into%20national%20strategies%20and%20programmes. 

World Health Organization. (n.d.). Maternal mortality. World Health Organization. https://www.who.int/news-room/fact-sheets/detail/maternal-mortality#:~:text=Key%20facts,dropped%20by%20about%2034%25%20worldwide. 

Zhao, D., Liu, D., Shi, W., Shan, L., Yue, W., Qu, P., Yin, C., & Mi, Y. (2023). Association between Maternal Blood Glucose Levels during Pregnancy and Birth Outcomes: A Birth Cohort Study. International journal of environmental research and public health, 20(3), 2102. https://doi.org/10.3390/ijerph20032102
## Appendix
##### Code for Data Exploration 
```
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

file = 'Maternal Health Risk Data Set.csv' 
data = pd.read_csv(file) 
data
data_cleaned = data.dropna()
output_file = 'data_cleaned.csv'
data_cleaned.to_csv(output_file, index=False) 
print(f"Cleaned data saved to '{output_file}'")
cleaned_file = 'data_cleaned.csv'
new_data = pd.read_csv(cleaned_file)
new_data
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

for i, (risk, group) in enumerate(new_data.groupby('RiskLevel')):
    axes[i].hist(group['Age'], bins=10, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Histogram - Age Distribution ({risk})')
    axes[i].set_xlabel('Age')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()

plt.show()
new_data.hist(figsize=(10, 8), bins=20)  
plt.show()
print(new_data.info())
print(new_data.describe())
print(new_data['RiskLevel'].value_counts())
#Correlation Matrix
# Convert 'RiskLevel' to numerical values
new_data['RiskLevel'] = new_data['RiskLevel'].map({'low risk': 0, 'mid risk': 1, 'high risk': 2}) 

#make correlation matrix 
correlation_matrix = new_data.corr()

# show correlation matrix 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()
#Summary statistics 1
age_column = 'Age'  

# Calculate the mean of the specified column
mean_value = new_data[age_column].mean()

print(f"The mean of the '{age_column}' column is: {mean_value}") 

median_value = new_data[age_column].median()

print(f"The median of the '{age_column}' column is: {median_value}")

#Summary statistics 2: low risk women  
low_risk_women = new_data[new_data['RiskLevel']=='low risk']
# mean age for women with low risk level
low_risk_mean_age = low_risk_women['Age'].mean()
print(f"The mean age of women with low risk level is: {low_risk_mean_age:.2f} years.")
# mean systolic BP  
low_risk_mean_sysBP = low_risk_women['SystolicBP'].mean()
print(f"The mean systolic blood pressure of women with low risk level is {low_risk_mean_sysBP:.2f} mmHg.")
# mean diastolic BP  
low_risk_mean_dsBP = low_risk_women['DiastolicBP'].mean()
print(f"The mean diastolic blood pressure of women with low risk level is {low_risk_mean_dsBP:.2f} mmHg.")
# mean BS
low_risk_mean_blood_sugar = low_risk_women['BS'].mean()
print(f"The mean blood sugar of women with low risk level is {low_risk_mean_blood_sugar:.2f} mmol/L.")

medium_risk_women = new_data[new_data['RiskLevel']=='mid risk']
#  the mean age for women with low risk level
medium_risk_mean_age = medium_risk_women['Age'].mean()
print(f"The mean age of women with medium risk level is: {medium_risk_mean_age:.2f} years.")
# mean systolic BP  
medium_risk_mean_sysBP = medium_risk_women['SystolicBP'].mean()
print(f"The mean systolic blood pressure of women with medium risk level is {medium_risk_mean_sysBP:.2f} mmHg.")
# mean diastolic BP  
medium_risk_mean_dsBP = medium_risk_women['DiastolicBP'].mean()
print(f"The mean diastolic blood pressure of women with medium risk level is {medium_risk_mean_dsBP:.2f} mmHg.")
# mean BS
medium_risk_mean_blood_sugar = medium_risk_women['BS'].mean()
print(f"The mean blood sugar of women with medium risk level is {medium_risk_mean_blood_sugar:.2f} mmol/L.")

# Summary statistics 3: medium risk women 
high_risk_women = new_data[new_data['RiskLevel']=='high risk']
# the mean age for women with low risk level
high_risk_mean_age = high_risk_women['Age'].mean()
print(f"The mean age of women with high risk level is: {high_risk_mean_age:.2f} years.")
# mean systolic BP  
high_risk_mean_sysBP = high_risk_women['SystolicBP'].mean()
print(f"The mean systolic blood pressure of women with high risk level is {high_risk_mean_sysBP:.2f} mmHg.")
# mean diastolic BP  
high_risk_mean_dsBP = high_risk_women['DiastolicBP'].mean()
print(f"The mean diastolic blood pressure of women with high risk level is {high_risk_mean_dsBP:.2f} mmHg.")
# mean BS
high_risk_mean_blood_sugar = high_risk_women['BS'].mean()
print(f"The mean blood sugar of women with high risk level is {high_risk_mean_blood_sugar:.2f} mmol/L.")

# Boxplot of blood sugar concentration at different levels 
risk_values_category = pd.concat([low_risk_women, medium_risk_women, high_risk_women])

plt.figure(figsize=(10, 6))
sns.boxplot(x='RiskLevel', y='BS', data=risk_values_category)
plt.title('Boxplot of Blood Sugar for Women in Different Risk Levels')
plt.show()

# Boxplot for systolic BP by risk level 
risk_values_category = pd.concat([low_risk_women, medium_risk_women, high_risk_women])

plt.figure(figsize=(10, 6))
sns.boxplot(x='RiskLevel', y='SystolicBP', data=risk_values_category)
plt.title('Boxplot of Systolic BP for Women in Different Risk Levels')
plt.show()

# Boxplot for diastolic BP by risk level 
risk_values_category = pd.concat([low_risk_women, medium_risk_women, high_risk_women])

plt.figure(figsize=(10, 6))
sns.boxplot(x='RiskLevel', y='DiastolicBP', data=risk_values_category)
plt.title('Boxplot of Diastolic BP for Women in Different Risk Levels')
plt.show()

# Boxplot for age in different risk levels 
risk_values_category = pd.concat([low_risk_women, medium_risk_women, high_risk_women])

plt.figure(figsize=(10, 6))
sns.boxplot(x='RiskLevel', y='Age', data=risk_values_category)
plt.title('Boxplot of Age for Women in Different Risk Levels')
plt.show()
```
##### Code for Random Forest Model 
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

data_frame = pd.read_csv('data_cleaned.csv')

label_enc = LabelEncoder()
data_frame['RiskLevel'] = label_enc.fit_transform(data_frame['RiskLevel'])

X = data_frame[['SystolicBP', 'DiastolicBP', 'HeartRate', 'Age', 'BS']]
y = data_frame['RiskLevel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

#training model 
rf_model.fit(X_train, y_train)

#save model with pickle file 
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_enc, le_file)

# get feature importance scores 
feature_importances = rf_model.feature_importances_

for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_importances, color='pink')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance for Random Forest Model')
plt.show()

#predictions
y_pred = rf_model.predict(X_test)

#convert risk level back to categorical values 
y_pred_original = label_enc.inverse_transform(y_pred)
y_test_original = label_enc.inverse_transform(y_test)

# check accuracy 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("Classification Report:")
print(classification_report(y_test_original, y_pred_original))

#confusion matrix 
cm = confusion_matrix(y_test_original, y_pred_original)
# order the categories 
class_order = ['low risk', 'mid risk', 'high risk']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_order, yticklabels=class_order)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

with open('random_forest_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_enc = pickle.load(le_file)

```
##### Code for XGBoost Model 
```
pip install xgboost

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb 
import pickle

data_frame = pd.read_csv('data_cleaned.csv')

# encoding risk level 
label_enc = LabelEncoder()
data_frame['RiskLevel'] = label_enc.fit_transform(data_frame['RiskLevel'])

# separating features from target variable 
X = data_frame[['SystolicBP', 'DiastolicBP', 'HeartRate', 'Age', 'BS']]
y = data_frame['RiskLevel']

# splitting the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Classifier 
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(data_frame['RiskLevel'].unique()), random_state=42)

# training the model 
xgb_model.fit(X_train, y_train) 

# save model as pickle file 
with open('xgboost_model.pkl', 'wb') as model_file:
    pickle.dump(xgb_model, model_file)

with open('label_encoder_xgb.pkl', 'wb') as le_file:
    pickle.dump(label_enc, le_file)

# predictions
y_pred = xgb_model.predict(X_test)

# converting risk levels back to categorical values 
y_pred_original = label_enc.inverse_transform(y_pred)
y_test_original = label_enc.inverse_transform(y_test)

# check accuracy 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("Classification Report:")
print(classification_report(y_test_original, y_pred_original))

# confusion matrix 
cm = confusion_matrix(y_test, y_pred)

class_order = ['low risk', 'mid risk', 'high risk']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_order, yticklabels=class_order)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

with open('xgboost_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

with open('label_encoder_xgb.pkl', 'rb') as le_file:
    label_enc_xgb = pickle.load(le_file)

```
#### Code for final_project.py
```
from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

app = Flask(__name__)

cleaned_final_data = pd.read_csv('data_cleaned.csv')

#Pickle file 
with open('xgboost_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# loading label encoder
with open('label_encoder_xgb.pkl', 'rb') as le_file:
    label_enc_xgb = pickle.load(le_file)

# Define acceptable ranges for input values
valid_ranges = {
    'age': (10, 100),
    'systolic_bp': (60, 190),
    'diastolic_bp': (30, 150),
    'heart_rate': (40, 150),
    'bs': (2, 20),
}

# for the homepage 
@app.route('/')
def index():
    return render_template('index.html')

# for the predictor page 
@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get values from physiological features
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])
        heart_rate = float(request.form['heart_rate'])
        age = float(request.form['age'])
        bs = float(request.form['bs'])

        # Validate input values
        validation_errors = []
        for name, value in {'age': age, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp, 'heart_rate': heart_rate, 'bs': bs}.items():
            min_val, max_val = valid_ranges[name]
            if not min_val <= value <= max_val:
                validation_errors.append(f"{name.capitalize()} should be between {min_val} and {max_val}.")

        # If there are errors in values, run error.html
        if validation_errors:
            return render_template('error.html', error_messages=validation_errors)

        # Create a dataframe with the values given by user 
        input_data = pd.DataFrame([[systolic_bp, diastolic_bp, heart_rate, age, bs]],
                                  columns=['SystolicBP', 'DiastolicBP', 'HeartRate', 'Age', 'BS'])   

        # Predict with model 
        prediction = xgb_model.predict(input_data)
        risk_level_xgb = label_enc_xgb.inverse_transform(prediction)[0]

        # Give the predictions and show them on result.html
        return render_template('result.html', risk_level=risk_level_xgb) 
    
# for visualizations page 
@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html') 
    
# for the contact page 
@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

if __name__ == "__main__":
    app.run(debug=True)

```
##### Code for index.html 
```
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to the HarmonyCare Maternal Health Predictor</title>

    <style>
        body {
            background-image: url(https://wallpapers.com/images/high/pregnancy-1200-x-700-background-cfk4e2bmc98xype3.webp);
            background-size: cover;
            text-align: center;
            color: #151414;
            padding: 50px;
            margin: 0; 
        }

        .header-box {
            background-color: rgba(221, 157, 212, 0.7); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
        }

        h1, h2, p {
            margin: 0;
        }

        h1 {
            font-size: 36px; 
        }

        h2 {
            font-size: 24px; 
        }

        p {
            font-size: 18px; 
        }

        a {
            color: #00ff00;
            text-decoration: none; 
        }

        nav {
            display: flex;
            justify-content: space-around;
            background-color: #333;
            padding: 10px 0;
        }

        nav a {
            color: #fff;
            text-decoration: none;
            font-weight: bold;
            font-size: 18px;
        }

        .content-box1 {
            background-color: rgba(252, 233, 63, 0.488); 
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(252, 233, 63, 0.488);
            text-align: center; 
            margin-bottom: 20px;
        }

        .content-box2 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        }

        .content-box3 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        }

        .content-box4 {
            background-color: rgba(253, 101, 202, 0.7);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 101, 202, 0.7);
            text-align: center; 
            margin-bottom: 20px;
        }
        .content-box4 p {
            font-size: 30px;
        }
        .content-box5 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        }
        .content-box6 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        } 
        .content-box8 {
            background-color: rgba(201, 250, 206, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(201, 250, 206, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        }
        .content-box9 {
            background-color: rgba(178, 157, 243, 0.693);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(178, 157, 243, 0.693);
            text-align: left; 
            margin-bottom: 20px;
        }

    </style>
</head>
<body>
    <div class="header-box">
        <h1>Welcome to the HarmonyCare Maternal Health Risk Predictor</h1>
        <h2>Prioritizing Health, Today and Tomorrow</h2>
    </div>

    <nav>
        <a href="/">Home</a>
        <a href="/predictor">Predictor</a>
        <a href="/visualizations">Data Visualizations</a>
        <a href="/contact_us">Contact Us</a>
    </nav>
    <br>
    <br>
    <br>
  
    <div class="content-box1">
        <h2>Did you know?</h2>
        <p>The WHO reports that in 2020, each day, almost <b>800 women</b> die from <u>preventable causes</u> related to pregnancy and childbirth.
            <br>
             A maternal death occurred almost <b><u>every two minutes</u></b> in 2020. 
            <br>
             Almost <b>95% of all maternal deaths</b> occurred in low and lower middle-income countries in 2020. 
        </p>
    </div> 
    
    <div class="content-box2">
        <p>Maternal mortality is an often-ignored but important issue throughout the world. The WHO lists the topic of maternal mortality as the third goal in the Sustainable Development Goals (SDG) and it aims to reduce global mortality rates to <b>less than 70 per 100,000 live births by 2030.</b></p>
    </div>

    <div class="content-box3">
        <p>While these goals have been stated, little progress has been made in terms of reducing mortality rates. On par with the SDGs of the WHO, the majority of the cases of maternal mortality are <b><u>PREVENTABLE</u></b>, therefore, I have created a maternal health risk predictor to determine a patient’s health risk in pregnancy based on several parameters. So:</p>
    </div> 

    <div class="content-box4">
        <p><b><u>Let’s get you started on your journey to good health for you and your little one.</u></b></p>
    </div> 

    <div class="content-box5">
        <h2>How does this work?</h2>
        <p>The HarmonyCare Maternal Health Risk Predictor is a machine learning model that uses several physiological factors such as age, systolic blood pressure, diastolic blood pressure, blood sugar concentration, and heart rate to determine a patient’s risk level for pregnancy.</p>
    </div> 

    <div class="content-box6">
        <h2>How will it be used?</h2>
        <p>The goal of the HarmonyCare Maternal Health Risk Predictor is to be used as a clinical decision support tool for physicians when assessing patients to give them a better understanding of the appropriate plan for treatment.</p>
    </div>  

    <div class="content-box8">
        <h2>What do the variables mean?</h2>
        <p>Apart from RiskLevel, the dataset has 6 features:
            Age is in years. Systolic blood pressure is in mmHg and diastolic blood pressure is in mmHg. The systolic blood pressure is the pressure in the arteries when the heart is 
            pumping and the diastolic blood pressure is the pressure in the arteries when the heart is at rest. The heart rate is in beats per minute, 
        the blood sugar concentration is measured by mmol/L, and the body temperature is in Fahrenheit. 
        </p>
    </div> 
    
    <div class="content-box9">
        <h2>Want more insights into the dataset and the model?</h2>
        <p>Here is the link to my <a href="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator"> Github repository.</a>
        </p>
    </div> 

    <footer>
        <p>&copy; 2023 HarmonyCare Maternal Health Risk Predictor</p>
    </footer>

</body>
</html>

```
##### Code for predictor.html 
```
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maternal Health Risk Predictor</title>

    <style>
        body {
            background-image: url(https://wallpapers.com/images/high/pregnancy-1000-x-600-background-f2fqthcgto21gd9h.webp);
            background-size: cover;
            background-position: center;
            font-family: 'Times New Roman', sans-serif;
            text-align: center;
            margin: 0;
            padding: 0;
        }
        .header-box {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin: auto;
            width: 50%;
            margin-top: 15px; 
        }

        nav {
            display: flex;
            justify-content: space-around;
            background-color: #333;
            padding: 10px 0;
            margin-top: 20px; 
        }

        nav a {
            color: #fff;
            text-decoration: none;
            font-weight: bold;
            font-size: 18px;
        }

        .container {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin: auto;
            width: 50%;
            margin-top: 20px; 
        }

        form {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin: 10px 0;
        }

        input {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            box-sizing: border-box;
        }

        button {
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        button:hover {
            background-color: #45a049;
        }
    </style>
</head>

<body>


    <div class="header-box">
        <h1>Maternal Health Risk Predictor</h1>
    </div>

    <nav>
        <a href="/">Home</a>
        <a href="/predict">Predictor</a>
        <a href="/visualizations">Data Visualizations</a>
        <a href="/contact_us">Contact Us</a>
    </nav>

    <div class="container">

        <h2>Please fill out the health information below:</h2>

        <form action="/predict" method="post">
            <label for="age">Age(years):</label>
            <input type="text" name="age" required>
            <br>
            <label for="bs">Blood Sugar(Bs)(mmol/L):</label>
            <input type="text" name="bs" required>
            <br>
            <label for="systolic_bp">Systolic BP(mmHg):</label>
            <input type="text" name="systolic_bp" required>
            <br>
            <label for="diastolic_bp">Diastolic BP(mmHg):</label>
            <input type="text" name="diastolic_bp" required>
            <br>
            <label for="heart_rate">Heart Rate(beats per minute):</label>
            <input type="text" name="heart_rate" required>
            <br>
            <button type="submit">Predict</button>
        </form>

        <p><a href="/">Back to Homepage</a></p>

    </div>

</body>

</html>

```
##### Code for visualizations.html 
```
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizations</title>

    <style>
        body { 
            background-size: cover;
            text-align: center;
            color: #121213fb;
            padding: 50px;
            margin: 0; 
        } 
        body {
            background-color: #fbc8f2; 
            padding: 50px;
            margin: 0;
        } 
        .header-box {
            background-color: rgba(221, 157, 212, 0.7); 
            padding: 20px;
            border-radius: 10px; 
            margin-bottom: 20px; 
        }

        h1, h2, p {
            margin: 0;
        }

        h1 {
            font-size: 36px; 
        }

        h2 {
            font-size: 24px;
        }

        p {
            font-size: 18px; 
        }

        a {
            color: #00ff00;
            text-decoration: none;
        }

        nav {
            display: flex;
            justify-content: space-around;
            background-color: #333;
            padding: 10px 0;
        }

        nav a {
            color: #fff;
            text-decoration: none;
            font-weight: bold;
            font-size: 18px;
        } 

        .content-box1 {
            background-color: rgba(253, 242, 252, 0.952); 
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: center; 
            margin-bottom: 20px;
        }

        .content-box2 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: center;
            margin-bottom: 20px;
        }

        .content-box3 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left;
            margin-bottom: 20px;
        }

        .content-box4 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: center;
            margin-bottom: 20px;
        }
        .content-box5 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: center;
            margin-bottom: 20px;
        }
        .content-box6 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: center; 
            margin-bottom: 20px;
        } 
        .content-box7 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: center; 
            margin-bottom: 20px;
        } 
        .content-box8 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        }
        .content-box9 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        }
        .content-box10 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        } 
        .content-box11 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left;
            margin-bottom: 20px;
        }
        .content-box12 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        }
        .content-box13 {
            background-color: rgba(253, 242, 252, 0.952);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(253, 242, 252, 0.952);
            text-align: left; 
            margin-bottom: 20px;
        }
        .content-box14 {
            background-color: rgba(99, 38, 243, 0.403);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(233, 157, 250, 0.545);
            text-align: left; 
            margin-bottom: 20px;
        }

        .graph-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap; 
        }

        .graph {
            flex: 1;
            margin: 10px;
        }
    
    </style>
</head>

<body>

<h1>Data Visualizations</h1>
<br>
<br>

<nav>
    <a href="/">Home</a>
    <a href="/predictor">Predictor</a>
    <a href="/visualizations">Data Visualizations</a>
    <a href="/contact_us">Contact Us</a>
</nav>
<br>
<br>

<div class="content-box1">
    <p>Here are the graphical representations that have been made to examine the dataset.</p>
</div>

<br> 

<div class="graph">
    <div class="content-box2">
    <h2>Data First Look</h2>
    <img src="{{ url_for('static', filename='graph_1.png') }}" alt="Graph 1" width="500" height="400">
    <p></p>
    </div>
</div> 
<div class="content-box13">
    <p>
        With these plots, it is observed that majority of the women in the dataset were between the ages of 15 and 35 years, 
        most of the blood sugar concentration levels were in between 6 and 8 mmol/L, 
        the heart rates were mostly between 62 to 80 bpm, and most of the women had a body 
        temperature of 98 degrees F. The systolic and diastolic blood pressure graphs are more spread out, with varied values. 
    </p>
    </div>
</div>

<div class="graph-container">
    <div class="content-box3">
        <h2>Boxplot of the Age of Women by Risk Level</h2>
        <img src="{{ url_for('static', filename='graph_5.png') }}" alt="Graph 1", width="500" height="400">
    </div> 

    <div class="content-box4">
        <h2>Boxplot of the Blood Sugar of Women by Risk Level</h2>
        <img src="{{ url_for('static', filename='graph_6.png') }}" alt="Graph 1", width="500" height="400">
    </div> 
</div> 
<div class="content-box12">
    <p> To check the spread of the data when it comes to ages in different risk level categories, 
        box plots were constructed for each risk level. The spread for each level shows that women at medium and high risk are women 
        who also fall between 30 and 50 years of age. There are outliers in the ages, since there are plenty of women that are in 
        their 60s and 70s and are still at low risk. There are some women who are medium risk who are in their 50s and 60s as well. 
    </p>
    <p>
        Research shows that increased blood sugar concentration of the mother during pregnancy can negatively affect 
        the health of the baby. Extreme cases of hyperglycemia(high blood glucose), can lead to gestational diabetes 
        which can affect the baby’s health and even the timing of the baby’s birth(Zhao et al., 2023). Therefore, box plots were 
        also made for the blood sugar concentrations based on risk level, which are shown in the image below. 
    </p>
</div>

<div class="content-box5">
        <h2>Correlation Matrix</h2>
        <img src="{{ url_for('static', filename='graph_7.png') }}" alt="Graph 1", width="500" height="400">
    </div>  
    <div class="content-box11">
        <p> In the matrix, each variable’s correlation to another variable is based on the scale from -1 to 1, with -1 
            indicating a strong negative correlation and +1 indicating a strong positive correlation. For age, 
            the strongest and most positive correlation is with blood sugar concentration, which has a correlation of 0.47 
            and vice versa. The strongest negative correlation for age is with body temperature, with a correlation of -0.26. 
            For systolic blood pressure, the strongest correlation, at 0.79, is with diastolic blood pressure and vice versa. 
            The strongest negative correlation for systolic blood pressure and diastolic blood pressure is with body temperature, 
            with a correlation of -0.29 and -0.26 respectively. For body temperature the strongest positive correlation is with 
            heart rate, at 0.10. 
        </p>
    </div>
<div class="content-box6">
        <h2>Feature Importances in Random Forest Model</h2>
        <img src="{{ url_for('static', filename='graph_8.png') }}" alt="Graph 1", width="500" height="400">
    </div> 
    <div class="content-box10">
        <p> The random forest model was trained and feature importances were extracted, 
            and the feature with the highest importance was blood sugar concentration, with a feature importance score of 0.369. 
        </p>
    </div>

<div class="graph-container">
    <div class="content-box7">
        <h2>Confusion Matrix for Random Forest Model</h2>
        <img src="{{ url_for('static', filename='graph_9.png') }}" alt="Graph 1", width="500" height="400">
    </div>
    <div class="content-box8">
        <h2>Confusion Matrix for XGBoost Model</h2>
        <img src="{{ url_for('static', filename='graph_10.png') }}" alt="Graph 1", width="500" height="400">
    </div> 
</div> 
<div class="content-box9">
    <p>Here are the confusion matrices of the random forest model(on the left) and the XGBoost model(to the right). For both models, 
        the risk levels were label encoded into numeric values. Then, the features were separated from the target variable, which 
        is the risk level, and then the models were trained and tested on a 80-20 split ratio. 
    </p> 
    <p>For the random forest model, The confusion matrix shows that 40 cases were correctly identified as low risk, 
        64 cases were correctly identified as mid risk, and 64 cases were correctly classified as high risk. 
        The total cases for the low risk level were 47, for the mid risk level were 80, and high risk level were 76. 
        The model predicted 46 out of the 47 cases to be low risk, 74 out of 80 cases to be mid risk, and 83 out of 76 cases as 
        high risk. 
    </p> 
    <p>An XGBoost classifier is run for the XGBoost model with multiclass softmax, since there are multiple risk levels in the dataset.
        In the confusion matrix, the model predicted 41 cases of high risk, 66 cases of low risk, and 64 cases of mid risk correctly. 
        The model predicted 48 out of the 47 cases to be low risk, 76 out of the 80 cases to be mid risk, and 79 out of the 76 cases 
        to be high risk.  
    </p>
</div>
<div class="content-box14">
    <h2>Want more insights into the dataset and the model?</h2>
    <p>Here is the link to my <a href="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator"> Github repository.</a>
    </p>
</div>
</body>
</html>  

```
##### Code for result.html
```
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maternal Health Risk Predictor - Result</title>
    <style>
        body {
            background-image: url(https://www.hopkinsmedicine.org/-/media/images/health/2_-treatment/gynecology/baby-holding-hands-with-mom-teaser.jpg); 
            background-size: cover;
            text-align: center;
            color: #0f0f0f; 
            padding: 50px;
            margin: 0; 
        }

        h1, p {
            margin: 0;
        }

        h1 {
            font-size: 36px;
        }

        p {
            font-size: 25px;
        }

        a {
            color: #441fec;
            text-decoration: none; 
        }
    </style>
</head>
<body>
    <h1>Maternal Health Risk Predictor - Result</h1>
    <p>The Predicted Risk Level is: <b>{{ risk_level }}</b></p> 

    <p><a href="/">Back to Homepage</a></p>
</body>
</html>
```
##### Code for error.html
```
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error</title> 
    <style>
        body { 
            background-image: url(https://png.pngtree.com/thumb_back/fh260/background/20231002/pngtree-d-rendering-of-error-cancel-ban-concept-circle-shaped-background-with-image_13543142.png);
            background-size: cover;
            text-align: center;
            color: #f8f8fafb;
            padding: 50px;
            margin: 0; 
        } 
    </style>
</head>
<body>
    <h1>Error</h1>
    {% for error_message in error_messages %}
        <p>{{ error_message }}</p>
    {% endfor %}
    <p><a href="/">Go back</a></p>
</body>
</html>
```
##### Code for contact_us.html 
```
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact Us</title>
    <style>
        body {
            background-image: url(https://www.shutterstock.com/image-illustration/website-internet-contact-us-icons-600nw-1537478306.jpg);
            background-size: cover;
            background-position: center;
            font-family: 'Arial', sans-serif;
            text-align: center;
            margin: 0;
            padding: 0;
            color: #fff; 
        }
        .header-box {
            background-color: rgba(170, 224, 247, 0.7); 
            padding: 10px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
        }

        .container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh; 
        }

        h1 {
            margin-bottom: 20px;
        }
        nav {
            display: flex;
            justify-content: space-around;
            background-color: #333;
            padding: 10px 0;
        }

        nav a {
            color: #fff;
            text-decoration: none;
            font-weight: bold;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <br>
    <br>
    <div class="header-box">
        <h1>Contact Us</h1>
    </div> 
    <nav>
        <a href="/">Home</a>
        <a href="/predictor">Predictor</a>
        <a href="/visualizations">Data Visualizations</a>
        <a href="/contact_us">Contact Us</a>
    </nav>
    <p>Feel free to reach out using the contact information below:</p>
    <p>Email: contact@harmonycare.com</p>
    <p>Phone: +123 456 789</p> 

</body>

</html> 
```

