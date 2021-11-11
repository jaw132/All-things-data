"""

Medical Transcription Dataset

Workflow:
1. Import data
2. Data exploration
3. Bias detection
4. Prepare and tranform data - feature engineering
5. Model train and tune - BERT


"""

## Importing data using ETL/Extracting/kaggle.sh

"""
 Data Exploration
"""

#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizer
from transformers import RobertaModel, RobertaConfig
from transformers import RobertaForSequenceClassification

# storing dataset in a pandas dataset to explore and basic cleaning
starting_dataframe = pd.read_csv("mtsamples.csv")

print('Shape of dataframe to start {}'.format(starting_dataframe.shape))

'''
df.isna().values.any() - checks if any entries in the dataframe are nan
df = df.reset_index(drop=True) - resets indexing 
e.g. 1  Dog
     2  Cat
     3  Mouse

will become 
     1 Dog
     2 Mouse

after removing cat and resetting the index, the drop option stops the old index
being added as a column
'''

# remove any nan from the dataframe
starting_dataframe = starting_dataframe.dropna()
starting_dataframe = starting_dataframe.reset_index(drop=True)
print('Shape of dataframe after removing NaNs {}'.format(starting_dataframe.shape))

# only interested in two columns - medical specialty, transcription
working_dataframe = starting_dataframe[["transcription", "medical_specialty"]]

# how many category are we dealing with
print('Number of medical specialties: {}'.format(len(starting_dataframe['medical_specialty'].unique())))

"""
Visualisation - Get a sense of how the data is distributed
"""

# plot a pie chart for the medical specialty

pie_chart = working_dataframe.groupby('medical_specialty').size().plot(kind='pie')
pie_chart.set_ylabel('Medical Specialty')

"""Can see surgery dominates, the rest are spread reasonably evenly. 
Might consider balancing the dataset"""

# same information in a bar chart, looks cleaner

specialty_plot = seaborn.countplot(working_dataframe["medical_specialty"])
specialty_plot.set_xticklabels(specialty_plot.get_xticklabels(), rotation=90)
specialty_plot.set_yticks(np.arange(0, 1100, 50))

# check the distribution of the transcriptions

transcription_lengths = [len(i) for i in working_dataframe["transcription"].tolist()]
seaborn.boxplot(transcription_lengths)

print('Max length of transcriptions is {}, minimum is {}'.format(max(transcription_lengths), min(transcription_lengths)))

"""The median length is just over 2000, but there are a significant 
amount which are quite a bit longer - might be important when deciding 
vector length """

"""
Feature Engineering

Need to make convert all the transcriptions to RoBERTa vectors

Also need to convert the medical specialties to numbers
"""

# basic map from categorical text to ints
def makeLabelsMap(series):
  unique_values = series.unique()
  labels_map, counter = {}, 0

  for value in unique_values:
    labels_map[value] = counter
    counter += 1

  return labels_map

# map the medical specialty to a number
labels_map = makeLabelsMap(working_dataframe["medical_specialty"])

working_dataframe["label_id"] = working_dataframe["medical_specialty"].map(labels_map)
print('Shape of dataframe with sentiment {}'.format(working_dataframe.shape))
working_dataframe.head()

# set up for the tokenisation of the transcriptions
# tokenization model
PRE_TRAINED_MODEL_NAME = 'roberta-base'

# max length of transcriptions - max length is 12000 greatest power of 2 under this is 8192
max_sequence_length = 512

# create the tokenizer to use based on pre trained model
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# Convert the review into the BERT input ids using 
def convert_to_bert_input_ids(transcription, max_seq_length):
    encode_plus = tokenizer.encode_plus(
          transcription,
          add_special_tokens=True,
          max_length=max_seq_length,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
    )

    return encode_plus['input_ids'].flatten().tolist()

working_dataframe['input_ids'] = working_dataframe['transcription'].apply(lambda transcription: convert_to_bert_input_ids(transcription, max_sequence_length))
print('working_dataframe[input_ids] after calling convert_to_bert_input_ids: {}'.format(working_dataframe['input_ids']))

# check what the dataframe is now looking like
working_dataframe.head()

# split into train, validation and test
train_percentage = 0.8
test_percentage = 0.1

holdout_percentage = 1 - train_percentage
print('holdout percentage {}'.format(holdout_percentage))
df_train, df_holdout = train_test_split(working_dataframe, 
                                        test_size=holdout_percentage) 
                                        

test_holdout_percentage = test_percentage / holdout_percentage
print('test holdout percentage {}'.format(test_holdout_percentage))
df_validation, df_test = train_test_split(df_holdout, 
                                          test_size=test_holdout_percentage)

df_train = df_train.reset_index(drop=True)
df_validation = df_validation.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print('Shape of train dataframe {}'.format(df_train.shape))
print('Shape of validation dataframe {}'.format(df_validation.shape))
print('Shape of test dataframe {}'.format(df_test.shape))

# storing the split data in csv files - best practise to store each transformation of the data - also means 
# you don't need to preprocess the data every time

df_train.to_csv('train_data.tsv', sep='\t', index=False)
df_validation.to_csv('validation_data.tsv', sep='\t', index=False)
df_test.to_csv('test_data.tsv', sep='\t', index=False)

# dataframe
df_train.head()   
df_validation.head()   
df_test.head()

"""
Training the Model
"""

# define the hyperparameters

TRAIN_BATCH_SIZE=32
    
VALIDATION_BATCH_SIZE=32

EPOCHS=3

FREEZE_BERT_LAYER=True
    
LEARNING_RATE=0.01

MOMENTUM=0.5  

SEED=42

LOG_INTERVAL=100
    
RUN_VALIDATION=False

# PyTorch dataset retrieves the dataset’s features and labels one sample at a time
# Create a custom Dataset class for the reviews
class ReviewDataset(Dataset):
    
    def __init__(self, input_ids_list, label_id_list):
        self.input_ids_list = input_ids_list
        self.label_id_list = label_id_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, item):
        # convert list of token_ids into an array of PyTorch LongTensors
        input_ids = self.input_ids_list[item] 
        label_id = self.label_id_list[item]

        input_ids_tensor = torch.LongTensor(input_ids)
        label_id_tensor = torch.tensor(label_id, dtype=torch.long)

        return input_ids_tensor, label_id_tensor

# PyTorch DataLoader helps to to organise the input training data in “minibatches” and reshuffle the data at every epoch
# It takes Dataset as an input
def create_data_loader(df, batch_size): 
    print("Get data loader")    
        
    ds = ReviewDataset(
        input_ids_list=df.input_ids.to_numpy(),
        label_id_list=df.label_id.to_numpy(),
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    ), df

# The RoBERTa model allows classification layers to be appended using 
# the RobertaConfig class see https://huggingface.co/transformers/model_doc/roberta.html
def configure_model():
    classes = starting_dataframe['medical_specialty'].unique()

    config = RobertaConfig.from_pretrained(
        PRE_TRAINED_MODEL_NAME, 
        num_labels=len(classes)
    )
    
    config.output_attentions=True

    return config

# function to train the model
def train_model(model,
                train_data_loader,
                val_data_loader):
    
    loss_function = nn.CrossEntropyLoss()    
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    # only want to train the final layer, so freeze the rest
    if FREEZE_BERT_LAYER:
        print('Freezing BERT base layers...')
        for name, param in model.named_parameters():
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False
        print('Set classifier layers to `param.requires_grad=False`.')        

    for epoch in range(EPOCHS):
        print('EPOCH -- {}'.format(epoch))

        for i, (sent, label) in enumerate(train_data_loader):
            model.train()
            optimizer.zero_grad()
            sent = sent.squeeze(0)
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()
            output = model(sent)[0]
            _, predicted = torch.max(output, 1)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
        
            # option to run validation after each epoch - to test accuracy improvements
            if RUN_VALIDATION:
                print('RUNNING VALIDATION:')
                correct = 0
                total = 0
                model.eval()

                for sent, label in val_data_loader:
                    sent = sent.squeeze(0)
                    if torch.cuda.is_available():
                        sent = sent.cuda()
                        label = label.cuda()
                    output = model(sent)[0]
                    _, predicted = torch.max(output.data, 1)

                    total += label.size(0)
                    correct += (predicted.cpu() ==label.cpu()).sum()

                accuracy = 100.00 * correct.numpy() / total
                print('[epoch/step: {0}/{1}] val_loss: {2:.2f} - val_acc: {3:.2f}%'.format(epoch, i, loss.item(), accuracy))
                       

    print('TRAINING COMPLETED.')
    return model

'''
MAIN PROCESSING - PART 1
'''

# Set up device to run on - use GPU if available
use_cuda = torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

device = torch.device('cuda' if use_cuda else 'cpu')


# Set the seed for generating random numbers

torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed(SEED) 

# Instantiate model

config = None
model = None

# Configure model
config = configure_model()
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    config=config
)

model.to(device)
    

if not model:
      print('Not properly initialized...')
else:
      print('Sucessfully downloaded model')

'''
MAIN PROCESSING - PART 2
'''

# Create data loaders

train_data_loader, df_train = create_data_loader(df_train, TRAIN_BATCH_SIZE)
val_data_loader, df_val = create_data_loader(df_validation, VALIDATION_BATCH_SIZE)

print("Processes {}/{} ({:.0f}%) of train data".format(
    len(train_data_loader.sampler), len(train_data_loader.dataset),
    100. * len(train_data_loader.sampler) / len(train_data_loader.dataset)
))

print("Processes {}/{} ({:.0f}%) of validation data".format(
    len(val_data_loader.sampler), len(val_data_loader.dataset),
    100. * len(val_data_loader.sampler) / len(val_data_loader.dataset)
)) 
    
print('model summary: {}'.format(model))


# Start training
model = train_model(
    model,
    train_data_loader,
    val_data_loader)

# check the validation accuracy after finishing training
correct = 0
total = 0
model.eval()

for sent, label in val_data_loader:
    sent = sent.squeeze(0)
    if torch.cuda.is_available():
        sent = sent.cuda()
        label = label.cuda()
    output = model(sent)[0]
    _, predicted = torch.max(output.data, 1)

    total += label.size(0)
    correct += (predicted.cpu() ==label.cpu()).sum()

accuracy = 100.00 * correct.numpy() / total

print(accuracy)