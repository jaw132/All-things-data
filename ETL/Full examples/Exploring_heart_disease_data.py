'''
Exploring a dataset heart disease, the dataset can be found at: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

The original dataset contained 76 attributes but it has been found the following 14 attribute yeild the best results:

1. age
2. sex
3. cp - chest pain type values (1 - typical angina, 2 - atypical angina, 3 - non-anginal pain, 4 - asymptomatic)
4. trestbps - resting blood pressure
5. chol - serum cholestoral in mg/dl
6. fbs - fasting blood sugar > 120 mg/dl (1 - True, 0 - False)
7. restecg - resting electrocardiographic results (0 - normal, 1- having ST-T wave abnormality, 2 - showing probable or definite left ventricular hypertrophy by Estes' criteria)
8. thalach - maximum heart rate
9. exang - exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest
11. slope - the slope of the peak exercise ST segment (1 - upsloping; 2 - flat; 3 - downsloping)
12. ca - number of major vessels (0-3) colored by flourosopy
13. thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
14. num (the predicted attribute), angiographic disease status based on the diameter narrowing


Note ST refers to a finding on an ECG.
'''

# -------------------------------------------------------
# extract and explore data
# -------------------------------------------------------

#import the libraries we will be using
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# lets read csv file into a dataframe, downloaded csv had no header
HD_dataframe = pd.read_csv('/content/processed.cleveland.csv', header=None,
                           names=["age", "sex", "chest pain", "rest bp", "cholestrol", "fasting blood sugar",
                                  "rest ECG", "max HR", "exer. ind. angina", "ST depression", "ST slope",
                                  "major vessels", "thal", "prediction"])
HD_dataframe.head()

# Noticed certain fields had values down as ? so remove all rows that contain one
question_mark_mask = (HD_dataframe != "?").all(1)

HD_dataframe = HD_dataframe[question_mark_mask]

# Also remove any nan from the dataframe
HD_dataframe = HD_dataframe.dropna()

# check the distribution of the data - skewed data will pose problems when it
# comes to statistical inference
HD_dataframe["prediction"].value_counts()

'''
So 54% so the data have no heart disease present, therefore we need a 
model that has at least 55% accuracy to be of any value. Otherwise 
predicting no heart disease every time is better (at least by this metric).
'''

# -------------------------------------------------------
# transform data and load into model
# -------------------------------------------------------

# split the dataframe into random train and test portions
train_df, test_df = train_test_split(HD_dataframe, test_size=0.15)

# separate the features from the labels
train_labels = train_df['prediction']
train_features = train_df.drop(columns=['prediction'])

test_labels = test_df['prediction']
test_labels_array = test_labels.to_numpy() # used in accuracy calcs.
test_features = test_df.drop(columns=['prediction'])


def cal_accuracy(predictions, actual):
    '''
    :param predictions: numpy array of model's predictions
    :param actual: numpy array of labels to compare preds with
    :return: percentages of predictions that are correct
    '''
    correct_sum = 0
    test_size = len(actual)
    for index in range(test_size):
        if actual[index] == predictions[index]:
            correct_sum += 1

    return correct_sum/test_size

# Instantiate a logistic regression model for multi variate classification
LR_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
# train on data
LR_model.fit(train_features.to_numpy(), train_labels.to_numpy())

# use the trained model to predict on the test set and check accuracy
lr_preds = LR_model.predict(test_features)
print(cal_accuracy(lr_preds, test_labels_array))

'''
Ending up being 66% accurate which is better than the 54% baseline and with 
such a small amount of data probably not that bad
'''




