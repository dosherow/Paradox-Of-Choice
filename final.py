import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import ConfusionMatrix
import os

os.chdir('/Users/drewosherow/Desktop/spring2019/itp449/python/')

# import data set and store in data frame
data = pd.read_csv('master_sheet.csv')
df = pd.DataFrame(data)

# use mean value of each column to fill null values (Drew)
df.fillna(df.mean(), inplace=True)
df.isnull().any()
print(df.isnull().any())
print(df.info())

# rename values of burgers to be one (Nigel)
df.dinner_choice.replace('Classic Hamburger', 'Classic Burger', inplace=True)

# checking for null values that might skew model (Drew)
null_values = df.isnull().sum()
null_values = null_values[null_values != 0].sort_values(ascending=False).reset_index()
null_values.columns = ['variable', 'number of missing']


# Data Visualization (Nigel, Drew, Philipp, Ailsa)
# Some General EDA
print(df.describe())
print(df.dtypes)
print(df.shape)

# How popular is each item in total
df.dinner_choice.value_counts().plot(kind='bar')
plt.xlabel('Type of Food')
plt.ylabel('Times Selected')
plt.title('Food Selection')
plt.show()

# Creating different data frames per survey
data_survey1 = df[df.Survey_no == 1]
data_survey2 = df[df.Survey_no == 2]
data_survey3 = df[df.Survey_no == 3]

# Histogram for 3 choice survey
data_survey1.choice_confidence.hist()
plt.title('Survey with 3 Choices Histogram')
plt.xlabel('Choice Confidence Selection')
plt.show()

# Histogram for 7 choice survey
data_survey2.choice_confidence.hist()
plt.title('Survey with 7 Choices Histogram')
plt.xlabel('Choice Confidence Selection')
plt.show()

# Histogram for 12 choice survey
data_survey3.choice_confidence.hist()
plt.title('Survey with 12 Choices Histogram')
plt.xlabel('Choice Confidence Selection')
plt.show()

# Corr Matrix Visualization for 3 Choice Survey
corr_matrix_1 = data_survey1.corr()
sns.heatmap(corr_matrix_1, annot=True)
plt.title('Correlation Matrix for 3 choices')
plt.show()

# Corr Matrix Visualization for 7 Choice Survey
corr_matrix_2 = data_survey2.corr()
sns.heatmap(corr_matrix_2, annot=True)
plt.title('Correlation Matrix for 7 choices')
plt.show()

# Corr Matrix Visualization for 12 Choice Survey
corr_matrix_3 = data_survey3.corr()
sns.heatmap(corr_matrix_3, annot=True)
plt.title('Correlation Matrix for 12 choices')
plt.show()

# Scatter plot between choice confidence and choice easiness
colors = {1: 'red', 2: 'blue', 3: 'green'}
df.plot(kind='scatter', x='choice_easiness', y='choice_confidence', c=df['Survey_no'].apply(lambda x: colors[x]))
plt.show()

# Box plot showing ranges of confidence for each number of choices
sns.boxplot(x=df['Survey_no'], y=df['choice_confidence'])
plt.show()


# Time Series Analysis (Gianna)
# convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# time of day
df['date'] = [d.date() for d in df['Timestamp']]
df['time'] = [d.time() for d in df['Timestamp']]

df['hour'] = df['time'].apply(lambda x: x.replace(minute=0, second=0))

# count per hour
countHour = df.groupby('hour')['age'].count()
print(countHour)

# Logistic Regression Model (Sophia, Sabrina, and Rachel)
# make new column for number of choices
df['num_choices'] = np.where(df['Survey_no'] == 1, '3 Choices',
                             np.where(df['Survey_no'] == 2, '7 Choices',
                                      '12 Choices'))

# make dummy variable for allergy column
df = pd.get_dummies(df, columns=['allergy'])

# define feature matrix and target variable
X = df[['choice_confidence', 'allergy_No']]
y = df['num_choices']

# split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = LogisticRegression()

# produce confusion matrix
cm = ConfusionMatrix(model)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)
cm.poof()

# calculate accuracy of model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# (Drew)
# dummy variables for categorical food data for prediction models to make numerical variables
one_hot = pd.get_dummies(df['dinner_choice'])
df = df.drop('Timestamp', axis= 1)
df = df.drop('age', axis = 1)
df = df.drop('date', axis = 1)
df = df.drop('time', axis = 1)
df = df.drop('hour', axis = 1)
df = df.drop('allergy_No', axis = 1)
df = df.drop('num_choices', axis = 1)
df = df.drop('dinner_choice', axis = 1)
df = df.join(one_hot)
pd.set_option('display.max_columns', 500)
print(df.info())
df[df.select_dtypes(['float']).columns] = df.select_dtypes(['float']).apply(lambda x: x.astype('int'))
df[df.select_dtypes(['uint8']).columns] = df.select_dtypes(['uint8']).apply(lambda x: x.astype('float'))
print(df.info())

# #split dataset into features and target variable
X1 = df[['choice_easiness', 'Blackened Steelhead Salmon', 'Buttermilk Fried Chicken',
        'Chicken Sugo Fettucine Pasta', 'Classic Burger', 'Classic Burger', 'Club Sandwich',
        'Cobb Salad', 'Grilled New Zealand Lamb Chop', 'Kale Caesar Salad', 'Local Fish Tacos',
         'Macaroni and Cheese', 'Margherita Pizza', 'Smoked Salmon Flatbread']]
y1 = df['choice_confidence']
# print(X1)
# print(y1)

#split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=80, max_features=4)
model.fit(X_train, y_train)
print("Random Forest Accuracy (Choice Confidence Target Variable):")
print(model.score(X_test,y_test))
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)
print(classification_report(y_test, y_predicted))

# running again but using choice_easiness as target variable

# #split dataset into features and target variable
X2 = df[['choice_confidence', 'Blackened Steelhead Salmon', 'Buttermilk Fried Chicken',
        'Chicken Sugo Fettucine Pasta', 'Classic Burger', 'Classic Burger', 'Club Sandwich',
        'Cobb Salad', 'Grilled New Zealand Lamb Chop', 'Kale Caesar Salad', 'Local Fish Tacos',
         'Macaroni and Cheese', 'Margherita Pizza', 'Smoked Salmon Flatbread']]
y2 = df['choice_easiness']
# print(X2)
# print(y2)

#split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = RandomForestClassifier(n_estimators=80, max_features=4)
model2.fit(X_train, y_train)
print("Random Forest Accuracy (Choice Easiness Target Variable):")
print(model2.score(X_test,y_test))
y_predicted2 = model2.predict(X_test)
cm2 = confusion_matrix(y_test, y_predicted2)
print(cm2)
print(classification_report(y_test, y_predicted2))


