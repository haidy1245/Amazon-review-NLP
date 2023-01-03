# -*- coding: utf-8 -*-
"""Amazon product review(NLP) (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kScu6Qiqmqp-0jGqCbxHE9kS6rzOsE0J
"""

# Commented out IPython magic to ensure Python compatibility.
#importing labraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

#read dataset
df= pd.read_csv("/content/Product_details.csv")
df

df['Product_Type'].value_counts()

df['Sentiment'].value_counts()

"""# Pre-Processing
1.Expanding Contraction
2.Language Detection
3.Tokenization
4.Converting all characters to lowercase
5.Removing Punctuation
6.Removing Stopwords
7.Lemmatization
"""

#importing requied labraries
import nltk
!pip install contractions
import contractions
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords , wordnet
from nltk.stem import WordNetLemmatizer
pd.set_option('display.max_colwidth',100)

# drop Text_ID column
df=df.drop('Text_ID',axis=1)

df

df.columns

# checking for null values
for col in df.columns:
    print(col,df[col].isnull().sum())

#Expanding Contractions
df['no_contract']=df['Product_Description'].apply(lambda x: [contractions.fix(word) for word in x.split()])
df

df.dtypes

#converting back to string
df['Product_Description_str']=[' '.join(map(str,l)) for l in df['no_contract']]
df

#English Language Detection
!pip install langdetect

from langdetect import detect

for sent in df['Product_Description_str']:
     df['lang']=detect(sent)
        
df

df['lang'].value_counts()

import nltk
nltk.download('punkt')
df['tokenized']=df['Product_Description_str'].apply(word_tokenize)
df

#Converting all the characters to Lowercase
df['lower']=df['tokenized'].apply(lambda x:[word.lower() for word in x])
df

#Removing Punctutions
punc= string.punctuation
df['no_punc']=df['lower'].apply(lambda x: [word for word in x if word not in punc])
df.head()

#Removing Stopwords
import nltk
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
df['no_stopword']=df['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])
df.head()

#Lemmatization
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
wnl=WordNetLemmatizer()
df['lemmatized']=df['no_stopword'].apply(lambda x:[wnl.lemmatize(word) for word in x])
df

"""# EDA(Exploratory Data Analysis)"""

!pip install pyLDAvis
import pyLDAvis.sklearn
from collections import Counter
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF 
from wordcloud import WordCloud, ImageColorGenerator

#converting back to string
df['lemmatized_str']=[' '.join(map(str,l)) for l in df['lemmatized']]
df

df1=df[['Product_Type','Sentiment','lemmatized_str','lemmatized']]
df1

#sentiment polarity analysis
df1['sentiment_polarity']=df1['lemmatized_str'].apply(lambda x: TextBlob(x).sentiment.polarity)
df1.head(20)

"""observation:-
    1.for sentiment=2 there is a negative and positive polarity 
    2.for sentiment=1 the prolarity is 1,+ve and also -ve.
    3.with above observation we can't say whether sentiment=1&2 contains positive words or negative. then it may neutral or mix of +ve& -ve sentiment.  
"""

pd.set_option('display.max_row',None)

df1['sentiment_polarity'].groupby(df1['Sentiment']).value_counts()

"""observation:-Each Sentiment has both Positive and Negative Polarity"""

df1['sentiment_polarity'].groupby(df1['Product_Type']).value_counts()

"""Only product 4 has positive sentiment polarity"""

#for the sentiment polarity we will plot a histogram and observe the distribution.
plt.figure(figsize=(30,20))
plt.xlabel('Sentiment Polarity',fontsize=50)
plt.xticks(fontsize=40)
plt.ylabel('Frequency',fontsize=20)
plt.yticks(fontsize=40)
plt.hist(df1['sentiment_polarity'],bins=50)
plt.title('Sentiment Distribution',fontsize=60)
plt.show()

"""Most of the Distribution lies on right side of 0.00 (i.e,Positive).So,overall we can conclude that the customers are happy with products"""

x=df1.Sentiment.value_counts()
y=x.sort_index()
plt.figure(figsize=(50,30))
sns.barplot(x.index,x.values,alpha=0.8)
plt.title("Sentiment Distribution",fontsize=40)
plt.ylabel('Frequency',fontsize=40)
plt.yticks(fontsize=40)
plt.xlabel('Sentiment',fontsize=40)
plt.xticks(fontsize=40)
plt.show()

df1[df1['Sentiment']==2]

plt.figure(figsize=(30,10))
plt.title('Percentage of Ratings',fontsize=20)
df1.Sentiment.value_counts().plot(kind='pie',labels=['Sentiment2','Sentiment3','Sentiment1','Sentiment0'],
                                 wedgeprops=dict(width=.7),autopct='%1.0f%%',startangle=-20,
                                 textprops={'fontsize':15})

df1.groupby('Sentiment')['sentiment_polarity'].mean().plot(kind='bar',figsize=(50,30))
plt.xlabel("Sentiment",fontsize=40)
plt.ylabel("Average Sentiment Polarity",fontsize=40)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('Avg. Sentiment polarity per Sentiment',fontsize=50)
plt.show()

df1['word_count']=df1['lemmatized'].apply(lambda x: len(str(x).split()))

df1

df1['word_count'].value_counts()

"""Let's observe that the longest review is for negative or neutral.this can be done by finding correlation matrix."""

df1.groupby('Sentiment')['word_count'].mean().plot(kind='bar',figsize=(50,30))
plt.xlabel('Sentiment',fontsize=40)
plt.ylabel('count of words',fontsize=40)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('Average No. of words per Sentiment Distribution',fontsize=40)
plt.show()

"""Sentiment 1 contains longest reviews"""

df1[df1['Sentiment']==1]

#correlation matrix
correlation=df1[['Sentiment','sentiment_polarity','word_count']].corr()
mask=np.zeros_like(correlation,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
plt.figure(figsize=(50,30))
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
sns.heatmap(correlation,cmap='coolwarm',annot=True,annot_kws={"size":40},linewidths=10,vmin=-1.5,mask=mask)

"""word_count and sentiment are negatively correlated"""

words= df1['lemmatized']
allwords=[]
for wordlist in words:
    allwords +=wordlist
    
print(len(allwords))    
print(allwords)

#wordcloud for top 100 most common words
mostcommon=FreqDist(allwords).most_common(100)

wordcloud=WordCloud(width=1600,height=800,background_color='white').generate(str(mostcommon))
fig=plt.figure(figsize=(40,15),facecolor='white')
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Top 100 Most Common Words',fontsize=100)

plt.tight_layout(pad=0)
plt.show()

"""this frequency analysis cenrtainly supports the overall positive sentiment of the reviews.
Terms such as good,love,awesome are showing that the customers are enjoying the products.

"""

#most frequent words for sentiment0.
#while interpreting the result for sentiment0 we have to be careful as it contribute only 2%(as shown is pie chart above)
group_by=df1.groupby('Sentiment')['lemmatized_str'].apply(lambda x: Counter(' '.join(x).split()).most_common(25))

#for sentiment0
group_by_0=group_by.iloc[0]
word0=list(zip(*group_by_0))[0]
freq0=list(zip(*group_by_0))[1]

plt.figure(figsize=(50,30))
plt.bar(word0,freq0)
plt.xlabel('Words',fontsize=50)
plt.ylabel('Frequancy of words',fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60,fontsize=40)
plt.title('Frequency of 25 words for sentiment0',fontsize=60)
plt.show()

"""we can remove the words such as sxsw, mention,google,link as they accure very frequenly.Also it is very difficult to drive insights from neutral words."""

#for sentiment1
group_by_1=group_by.iloc[1]
word0=list(zip(*group_by_0))[0]
freq0=list(zip(*group_by_0))[1]

plt.figure(figsize=(50,30))
plt.bar(word0,freq0)
plt.xlabel('Words',fontsize=50)
plt.ylabel('Frequancy of words',fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60,fontsize=40)
plt.title('Frequency of 25 words for sentiment1',fontsize=60)
plt.show()

"""sentiment1 also don't have clues for product improvement."""

#for sentiment2
group_by_2=group_by.iloc[2]
word0=list(zip(*group_by_0))[0]
freq0=list(zip(*group_by_0))[1]

plt.figure(figsize=(50,30))
plt.bar(word0,freq0)
plt.xlabel('Words',fontsize=50)
plt.ylabel('Frequancy of words',fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60,fontsize=40)
plt.title('Frequency of 25 words for sentiment2',fontsize=60)
plt.show()

group_by_3=group_by.iloc[3]
word0=list(zip(*group_by_0))[0]
freq0=list(zip(*group_by_0))[1]

plt.figure(figsize=(50,30))
plt.bar(word0,freq0)
plt.xlabel('Words',fontsize=50)
plt.ylabel('Frequancy of words',fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60,fontsize=40)
plt.title('Frequency of 25 words for sentiment3',fontsize=60)
plt.show()

df.head()

"""same 25 words are repeating for all the sentiments.there is no such words which shows positiveness. therefore we can't conclude anything from these words.it is better to remove them all.

# Topic Modeling

# Count Vectorizer
"""

df2=df1[['Sentiment','Product_Type','sentiment_polarity']]
df2

#Count Vectorizer
tf_vectorizer = CountVectorizer(max_df=0.9,min_df=25,max_features=6000)
tf=tf_vectorizer.fit_transform(df1['lemmatized_str'].values.astype('U'))
tf_feature_names = tf_vectorizer.get_feature_names()

df3 = pd.DataFrame(tf.toarray(),columns=list(tf_feature_names))
df3

d=[df2,df3]
data=pd.concat(d,axis=1)
data

x=data.drop('Sentiment',axis=1)
y=data['Sentiment']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)

"""# Model Building(Naive Bayes)"""

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

#Accuracy Check
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)

accuracy_score(y_test,y_pred)



#without product type and sentiment polarity
df2=df1['Sentiment']
d=[df2,df3]
data1=pd.concat(d,axis=1)
data1

X=data1.drop('Sentiment',axis=1)
Y=data1['Sentiment']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=3)

classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
accuracy_score(Y_test,Y_pred)



#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

from sklearn import metrics
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))





#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)

from sklearn import metrics
print('Accuracy:',metrics.accuracy_score(Y_test,Y_pred))

from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

#TF-IDF
#instead of using count vectorizer, now we will use TF-IDF. this method help to bring down the weight of high frequency words.
tfidf_vectorizer = TfidfVectorizer(max_df=0.90,min_df=25,max_features=6000,use_idf=True)

tfidf=tfidf_vectorizer.fit_transform(df1['lemmatized_str'])
tfidf_feature_names=tfidf_vectorizer.get_feature_names()

df3=pd.DataFrame(tfidf.toarray(),columns=list(tfidf_feature_names))
df3

d=[df2,df3]
data1=pd.concat(d,axis=1)
data1

X=data1.drop('Sentiment',axis=1)
Y=data1['Sentiment']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

#Naive Bayes model
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,Y_train)

#predict and accuracy
y_pred1=classifier.predict(X_test)

#Accuracy
accuracy_score(Y_test,y_pred1)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

from sklearn import metrics
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))

from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

model = RandomForestClassifier

model= RandomForestClassifier()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print(classification_report(y_test,y_pred))

import pickle
pickle.dump(model, open('rf.pkl', 'wb'))

