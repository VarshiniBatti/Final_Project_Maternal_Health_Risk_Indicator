# Final Project Report 
## Introduction
The topic that was chosen to explore in the final project is maternal health. According to WHO, in 2020, almost 800 women died each day from completely preventable causes that are related to pregnancy and childbirth(World Health Organization, n.d.). Specifically, in the same year, approximately 95% of all maternal deaths occurred in low and middle income countries(World Health Organization, n.d.). The UN lists the topic of maternal mortality as the third goal in the Sustainable Development Goals (SDG) and it aims to reduce global mortality rates to less than 70 per 100,000 live births by 2030(UN Women, n.d.). While these goals have been stated, little progress has been made in terms of actually reducing mortality rates. Maternal health and mortality is a very significant matter, however, it is seldom discussed when talking about existent issues in the world, even though it is preventable. Therefore, in order to shed light on this issue, the aim of the project was to develop a machine learning model that predicts the intensity of health risk for patients that are going through pregnancy using several biological and physiological variables. 

The dataset that has been used for the project is from Kaggle, and has originally been collected by researchers investigating health risk factors for pregnant women in rural areas of Bangladesh through an IoT based maternal healthcare system, with sensors for heart rate, blood pressure, blood glucose levels, and other variables. The dataset was collected from the IoT device, local hospitals, and web portals in Bangladesh, and stored in a csv file. For this project, the csv file was acquired using a file download. The data set was interesting because it had recorded distinct physiological features that are commonly measured to predict the risk level of the mother. 

The data provider, Kaggle, was in compliance with the FAIR principles. The dataset was findable for many reasons. Since there were different physiological variables present, such as blood sugar concentration, blood pressure, body temperature, and heart rate, the data provider clearly specified the metadata and the units used for each factor clearly on the page. The dataset did not contain unique identifiers for patients, and just listed out their values without any identifying personal information. The dataset is easily searchable through Kaggle. The dataset is also accessible because in Kaggle, the dataset is under the Attribution 4.0 International license by Creative Commons, which allows for sharing and adapting the dataset to derive insights with open access as long as the appropriate credit is given to the authors. The dataset is from the UC Irvine Machine Learning Repository, and is from a research paper investigating maternal health risk factors. The dataset is in the form of a CSV file. The dataset is interoperable because it uses the same units for all of its factors. Most of the factors are represented in SI units, for example blood sugar concentration is in mmol/L, and heart rate is in beats per minute. The Creative Commons license is clear in that it states that it allows for the dataset to be shared, copied, transformed, and adapted, therefore it is also reusable. The dataset was well-annotated with metadata and the licensing terms were clear. 
## Data Exploration 
The dataset contained 1014 rows and 7 columns, which are age in years, systolic and diastolic blood pressure in mmHg, body temperature in Fahrenheit, blood sugar concentration(represented as BS) in mmol/L, and heart rate in beats per minute. The dataset did not have NA values. Therefore, Then, in order to look at the distribution and the range of data, histograms were plotted for each column, whose images are shown below: 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_1.png" alt="" width="400" height="400">

With these plots, it is observed that majority of the women in the dataset were between the ages of 15 and 35 years, most of the blood sugar concentration levels were in between 6 and 8 mmol/L, the heart rates were mostly between 62 to 80 bpm, and most of the women had a body temperature of 98 degrees F. The systolic and diastolic blood pressure graphs are more spread out, with varied values. According to the CDC, the normal values for the systolic blood pressure are around 120 mmHg and for the diastolic pressure are around 80 mmHg. The patient is considered to have elevated blood pressure of hypertension if their systolic blood pressure is greater than 130 mmHg and if their diastolic blood pressure is greater than 80 mmHg(cite). To delve deeper, the dataset was then filtered by risk level to look at the distribution of data by risk level. The age distribution per risk level can be observed in the graphs below. 

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
The second machine learning model that was run on the cleaned dataset was the XGBoost model. The target variable, risk level, was label encoded into numerical values in the same manner as with the Random Forest model. Then the features were separated from Risk Level and the data was split into training and testing sets of 80-20 ratio. An XGBoost classifier is run with multiclass softmax, since there are multiple risk levels in the dataset. The model was then trained and tested, and the accuracy of the model came out to be 0.84. Classification reports were used to validate the analyses in the XGBoost model as well.  In the report, the F1 scores for high risk were 0.86, for low risk were 0.85, and for mid risk were 0.83. This means that the XGBoost model was slightly better at classifying the correct levels of risk for patients and minimized a large amount of false positives as well. In the confusion matrix, the model predicted 41 cases of high risk, 66 cases of low risk, and 64 cases of mid risk correctly. The model predicted 48 out of the 47 cases to be low risk, 76 out of the 80 cases to be mid risk, and 79 out of the 76 cases to be high risk. Overall, the values from the F1 scores and from the confusion matrix indicate that the XGBoost model was a more accurate predictor of maternal health risk level compared to Random Forest. Therefore, the XGBoost model was the final model to be deployed for the website of the predictor tool. Both the classification report and the confusion matrix for the Random Forest model are shown below. 

Classification Report for XGBoost Model 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_14.png" alt="" width="400" height="400">

Confusion Matrix for XGBoost Model 

<img src="https://github.com/VarshiniBatti/Final_Project_Maternal_Health_Risk_Indicator/blob/main/templates/static/graph_10.png" alt="" width="400" height="400">

## Server API and Web Interface 
The goal of the website was to introduce the predictive model in a user-friendly manner, while providing more information about the features, and the general issue about maternal health and maternal mortality. The website has four pages: the homepage, the predictor tool, the visualizations, and a contact page. The homepage consists of general statistics and information about maternal health, a link to the Github repository for all the files of the project, and a navigation bar that shows all the webpages. The predictor model is another webpage on the website, and it takes in the age, systolic and diastolic blood pressure, heart rate, and blood sugar concentration of the patient and returns the risk level of the patient as a result page. The prediction also has specific boundaries for the values of each input and if the values are outside those ranges, then it returns an error message. Another webpage on the website is the data visualizations page, which has all the visualizations that have been done in the project. Lastly, to make the website mimic a real-world website user design interface, there is also a contact page for users as well. 
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

##### Code for Random Forest Model 
##### Code for XGBoost Model 
#### Code for final_project.py
##### Code for index.html 
##### Code for predictor.html 
##### Code for visualizations.html 
##### Code for result.html
##### Code for error.html
##### Code for contact_us.html 

