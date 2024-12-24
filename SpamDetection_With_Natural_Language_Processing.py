# %%
#installing the necessary libraries for analysis and downloading the NLTK shell
import nltk; import pandas as pd 
nltk.download('stopwords')

# %%
# nltk.download_shell()

# %%
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Authenticate with Kaggle
api = KaggleApi()
api.authenticate()

# Define dataset information
dataset = "nittalasatyasrinivas/smsspamcollectiontsv"
file_name = "SMSSpamCollection.tsv"

# Download the dataset to a specified location
destination = "data/"  # Choose your desired folder
os.makedirs(destination, exist_ok=True)
api.dataset_download_file(dataset, file_name, path=destination)

# Extract the file if downloaded as a zip
zip_path = os.path.join(destination, file_name + ".zip")
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    os.remove(zip_path)

# Load the file into a pandas DataFrame
file_path = os.path.join(destination, file_name)
if os.path.exists(file_path):
    # Read the data into a DataFrame
    messages = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
    
    # Output the first few rows
    print("First 5 rows of the dataset:")
    print(messages.head())
    
    # Print first 10 messages with their index
    print("\nFirst 10 messages:")
    for mess_no, message in enumerate(messages['message'][:10]):
        print(f"{mess_no}: {message}\n")
else:
    print(f"File not found at {file_path}.")

# %%
#overview of the dataframe
messages.describe()

# %%
#looking at the number of spam and non-spam messages, number of unique messages and most frequent messages
messages.groupby('label').describe()

# %%
#adding a calculated field that shows the length of messages
messages['length'] = messages['message'].apply(len)
messages.head()

# %%
#some libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sb

# %%
#histogram showing the distribution of message length
messages['length'].plot(kind='hist',bins=100)

# %%
#some statitics to guide our analysis
messages['length'].describe()

# %%
#looking at the outliers (longest message)
messages[messages['length']==910]['message'].iloc[0]

# %%
#different distributions of spam and non-spam messages
messages.hist(column='length',by='label',bins=60)

# %%
#prototyping the string library with a sample text message
import string
mess= 'Sample string: Notice, it has punctuation!'
string.punctuation

# %%
#vectorizing our prototype as per TF-IDF conventions
no_punc = [c for c in mess if c not in string.punctuation]
no_punc

# %%
no_punc=''.join(no_punc) 
#the blank space ('') here is the delimeter were choosing to create a sentence from the vectorized string above
no_punc

# %%
from nltk.corpus import stopwords
#filtering out stopwords to get unique words
# stopwords.words('english')

# %%
no_punc=no_punc.split()

# %%
clean_mess=[word for word in no_punc if word not in stopwords.words('english')]

# %%
clean_mess

# %%
#defining a data wrangling function


def text_process(mess):
    """
    1. remove punctuation
    2.remove stopwords
    3.return list of clean text
    
    """
    no_punc = [c for c in mess if mess not in string.punctuation]
    
    no_punc = ''.join(no_punc)
    
    clean_mess = [word for word in no_punc.split() if word.lower not in stopwords.words('english')]
    
    return (clean_mess)

# %%
# Now to convert each message, represented as a list of tokens above, 
# into a vector that machine learning models can understand.
# This is done in three steps using the bag-of-words model:
# 1. Count how many times does a word occur in each message (Known as term frequency)
# 2. Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
# 3. Normalize the vectors to unit length, to abstract from the original text length (L2 norm)

from sklearn.feature_extraction.text import CountVectorizer
bow_tranformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# %%
#testing whether the vectorization works
mess4=messages['message'][3]
mess4

# %%
#the vectorized bag of words
bow4= bow_tranformer.transform([mess4])
print(bow4)

# %%
#the shape of our bag of words
print(bow4.shape)

# %%
#get_feature_names is a function of sci-kit learn's feature_extraction that lets you see the actual value of the
#vectorized word
bow_tranformer.get_feature_names()[5205]

# %%
#transforming our bag of words into as sparse matrix for correlation and classification 
messages_bow = bow_tranformer.transform(messages['message'])

# %%
print('The shape of the sparse matrix is ', messages_bow.shape)

# %%
messages_bow.nnz #non-zero occurences

# %%
from sklearn.feature_extraction.text import TfidfTransformer
#using the statistical approach of TF-IDF which is a fundamental technique of NLP

# %%
TfidfTransf=TfidfTransformer().fit(messages_bow)

# %%
Tfid4 = TfidfTransf.transform(bow4)

# %%
#each word and its associated Inverse Document Frequency
print(Tfid4)

# %%
#testing the IDF (inverse document frequency) of the word `"u"` and of word `"university"`?
print(TfidfTransf.idf_[bow_tranformer.vocabulary_['u']])
print(TfidfTransf.idf_[bow_tranformer.vocabulary_['university']])

# %%
#transforming the entire bag-of-words corpus into TF-IDF corpus at once:
messages_tfidf = TfidfTransf.transform(messages_bow)
print(messages_tfidf.shape)

# %%
#building and training a model for classifaction
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

# %%
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)

# %%
#We can use SciKit Learn's built-in classification report, which returns [precision, recall,f1-score] to test accuracy
from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))

# %%
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

# %%
# Using SciKit Learn's capabilities to store a pipeline of workflow. 
# allowing the set up of all the transformations that we will do to the data for future use. 

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# %%
# Now I can directly pass message text data and the pipeline will do all the pre-processing.
# We can treat it as a model/estimator API:
pipeline.fit(msg_train,label_train)

# %%
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))

# %%
