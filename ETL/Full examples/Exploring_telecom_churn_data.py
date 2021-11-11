'''
Exploring a telecom churn dataset with the idea to predict whether
a customer would leave or not based factors like how long they have been with
a sevice provider, customer service calls etc.
'''

#import the libraries we will be using
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------------------------------------
# explore and transform data
# ------------------------------------------------------------

# take an initial look at data
df = pd.read_csv('/content/telecom_churn.csv')
df.head()

# check the shape of the data
df.shape

df.info() # this will give all the columns and what type they are

# change text data to numerical
int_map = {'No' : 0, 'Yes' : 1}
df['International plan'] = df['International plan'].map(int_map)
df['Voice mail plan'] = df['Voice mail plan'].map(int_map)

#remove state as it adds to much noise
df = df.drop(columns=['State'])

# finally let's change the predicted value to numeric
df['Churn'] = df['Churn'].astype('int')

# ------------------------------------------------------------
# split data and load into model
# ------------------------------------------------------------

# split the data into train and test
train_churn, test_churn = train_test_split(df, test_size=0.2)
train_churn_l = train_churn['Churn']
train_churn_f = train_churn.drop(columns=['Churn'])

test_churn_l = test_churn['Churn']
test_churn_f = test_churn.drop(columns=['Churn'])


# Instantiate a k nearest neighbours model and train with data
knn_model = KNeighborsClassifier(11)
knn_model.fit(train_churn_f, train_churn_l)

# predict the test data
knn_preds = knn_model.predict(test_churn_f)


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


# print out accuarcy of model
print(cal_accuracy(knn_preds, test_churn_l.to_numpy()))

