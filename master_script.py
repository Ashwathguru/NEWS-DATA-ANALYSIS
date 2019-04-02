#!/usr/bin/env python
# coding: utf-8

# In[3]:


#DATA ENGINEERING
#IMPORT LIBS
from pandas.io.json import json_normalize
import json
import pandas as pd
import numpy as np
import datetime
from datetime import date
import calendar


# In[2]:


#COPIES THE CONTENT OF A JSON FILE INTO A DATAFRAME
data = pd.read_json('News_Category_Dataset.json', lines=True)


# In[1]:


idx=list(np.arange(1,200854))
date_list=list(data['date'])
author_list=list(data['authors'])
category_list=list(data['category'])
headline_list=list(data['headline'])
link_list=list(data['link'])
short_list=list(data['short_description'])


# In[ ]:


#EXTRACT EXTRA FEATURES LIKE DAY OF THE WEEK FROM EXISTING DATA AND APPEND ALL THE DATA TO A CSV
day=[]
weekday=[]
month_list=[]
year_list=[]

for i in date_list:
    d_name=calendar.day_name[i.weekday()]
    weekday.append(d_name)
    datee = datetime.datetime.strptime(str(i), "%Y-%m-%d %H:%M:%S")

    m=datee.month
    month_list.append(m)
    y=datee.year
    year_list.append(y)
    d=datee.day
    day.append(d)

clean_data=pd.DataFrame({'SERIAL_NUMBER':idx,'DATE':date_list,'YEAR':year_list,'MONTH':month_list,'DAY':day,'DAY_OF_WEEK':weekday,'AUTHOR':author_list,'CATEGORY':category_list,'HEADLINE':headline_list,'LINK':link_list,'SHORT_DESC':short_list})


# In[ ]:


clean_data.to_csv('master_data.csv',sep=',',header=True,index=False)


# In[4]:


#DATA ANALYSIS
#IMPORT LIBS
import re
import pandas as pd # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
import numpy as np
import sklearn
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[5]:


master_data = pd.read_csv("master_data.csv")
master_data['AUTHOR'] = master_data['AUTHOR'].fillna('None')
#print(news)
#[	SERIAL_NUMBER	DATE	YEAR	MONTH	DAY	DAY_OF_WEEK	AUTHOR	CATEGORY	HEADLINE	LINK	SHORT_DESC]


# In[9]:


#FUNCTION TO CROP DATA BETWEEN TWO GIVEN DATES
def data_slicer(start_date,end_date):
    index = pd.date_range(start=start_date,end=end_date)
    date_list=[]
    for ch in index:
        datee = datetime.datetime.strptime(str(ch), "%Y-%m-%d %H:%M:%S")
        datee2=datee.strftime("%d-%m-%Y")
        date_list.append(datee2)
    data_req=master_data[master_data['DATE'].isin(date_list)]
    return data_req

print(data_slicer('26-06-2012','31-12-2012'))


# In[28]:


#GETS A PARTICULAR MONTHS DATA IN ALL 6 years
def month_data(month_number):
    data_req=master_data[master_data['MONTH'] == month_number]
    return data_req
print(month_data(6))


# In[29]:


#FUNCTION TO GET A PARTICULAR DAY OF THE WEEK DATA FROM ALL 6 YEARS
def day_of_week(day_name):
    data_req=master_data[master_data['DAY_OF_WEEK'] == day_name]
    return data_req

print(day_of_week('Monday'))


# In[30]:


#GET AUTHOR BASED ON NAME SEARCH
def data_author(author_name):
    #data_req=master_data[master_data['AUTHOR'] == author_name]
    data_req=master_data[master_data.AUTHOR.str.contains(author_name)]
    return data_req

print(data_author('Ron'))


# In[13]:


#DATE TO DATE REPORTS
def date_to_date_analysis(start_date,end_date):
    data=data_slicer(start_date,end_date)
    category_count={}
    author_article_count={}
    year_list=list(set(list(data['YEAR'])))
    category_list=list(set(list(data['CATEGORY'])))
    authors_list=list(set(list(data['AUTHOR'])))
    print("\nThe active year/s in the given range :",year_list)
    print("\nThe categories published :",category_list)
    print("\nThe number of active authors :",len(authors_list))
    for cat in category_list:
        data1=data[data['CATEGORY']==cat]
        category_count[cat] = data1['CATEGORY'].count()
    print(category_count)
    
    for cat in authors_list:
        data1=data[data['AUTHOR']==cat]
        author_article_count[cat] = data1['AUTHOR'].count()
    #print(author_article_count)
    
    high_auth=[]
    highest_count=0
    highest_name='AA'
    for name,count in author_article_count.items():
        if count>highest_count:
            highest_count=count
            highest_name=name
        if count > 200:
            high_auth.append(name)
    print("\nAuthors who wrote more than 200 posts:",high_auth)
    print("\nMost Active Author in the given date range :",highest_name,highest_count)

date_to_date_analysis('26-05-2016','29-06-2017')


# In[14]:


master_data.groupby(by='CATEGORY').size()


# In[9]:


#count the number of author in the dataset
#news.authors.value_counts()
total_authors = master_data.AUTHOR.nunique()
news_counts = master_data.shape[0]
print('Total Number of authors : ', total_authors)
print('avg articles written by per author: ' + str(news_counts//total_authors))
print('Total news counts : ' + str(news_counts))


# In[10]:


#DERIVING INFERENCES
authors_news_counts = master_data.AUTHOR.value_counts()
sum_contribution = 0
author_count = 0
for author_contribution in authors_news_counts:
    author_count += 1
    if author_contribution < 80:
        break
    sum_contribution += author_contribution
print('{} of news is contributed by {} authors i.e  {} % of news is contributed by {} % of authors'.
      format(sum_contribution, author_count, format((sum_contribution*100/news_counts), '.2f'), format((author_count*100/total_authors), '.2f')))


# In[11]:


master_data.AUTHOR.value_counts()[0:10]


# In[25]:


author_name = 'Lee Moran'
#author_name = 'Ed Mazza'
particular_author_news = master_data[master_data['AUTHOR'] == author_name]
df = particular_author_news.groupby(by='CATEGORY')['HEADLINE'].count()
df


# In[13]:


#DATA VISUALIZATION
fig, ax = plt.subplots(1, 1, figsize=(35,7))
sns.countplot(x = 'CATEGORY', data = master_data)


# In[14]:


fig, ax = plt.subplots(1, 1, figsize=(15,15))
master_data['CATEGORY'].value_counts().plot.pie( autopct = '%1.1f%%')


# In[11]:


data_2012=data_slicer('26-06-2012','31-12-2012')
data_2013=data_slicer('01-01-2013','31-12-2013')
data_2014=data_slicer('01-06-2014','31-12-2014')
data_2015=data_slicer('01-06-2015','31-12-2015')
data_2016=data_slicer('01-06-2016','31-12-2016')
data_2017=data_slicer('01-06-2017','31-12-2017')
data_2018=data_slicer('01-06-2018','26-05-2018')


# In[113]:


#2012
fig, ax = plt.subplots(1, 1, figsize=(10,10))
data_2012['CATEGORY'].value_counts().plot.pie( autopct = '%1.1f%%')
#MOST ACTIVE AUTHOR
x12=data_2012['AUTHOR'].value_counts()
print("Top 5 authors in 2012 \n",x12[:6])


# In[115]:



#2013
fig, ax = plt.subplots(1, 1, figsize=(10,10))
data_2013['CATEGORY'].value_counts().plot.pie( autopct = '%1.1f%%')
x13=data_2013['AUTHOR'].value_counts()
print("\nTop 5 authors in 2013 \n",x13[:6])


# In[116]:


#2014
fig, ax = plt.subplots(1, 1, figsize=(8,8))
data_2014['CATEGORY'].value_counts().plot.pie( autopct = '%1.1f%%')
x14=data_2014['AUTHOR'].value_counts()
print("\nTop 5 authors in 2014 \n",x14[:6])


# In[117]:


#2015
fig, ax = plt.subplots(1, 1, figsize=(8,8))
data_2015['CATEGORY'].value_counts().plot.pie( autopct = '%1.1f%%')
x15=data_2015['AUTHOR'].value_counts()
print("\nTop 5 authors in 2015 \n",x15[:6])


# In[ ]:





# In[118]:


#2016
fig, ax = plt.subplots(1, 1, figsize=(8,8))
data_2016['CATEGORY'].value_counts().plot.pie( autopct = '%1.1f%%')
x16=data_2016['AUTHOR'].value_counts()
print("\nTop 5 authors in 2016 \n",x16[:6])


# In[119]:


#2017
fig, ax = plt.subplots(1, 1, figsize=(8,8))
data_2017['CATEGORY'].value_counts().plot.pie( autopct = '%1.1f%%')
x17=data_2017['AUTHOR'].value_counts()
print("\nTop 5 authors in 2017 \n",x17[:6])


# In[120]:


#2018
fig, ax = plt.subplots(1, 1, figsize=(8,8))
data_2018['CATEGORY'].value_counts().plot.pie( autopct = '%1.1f%%')

x18=data_2018['AUTHOR'].value_counts()
print("\nTop 5 authors in 2018 \n",x18[:6])


# In[59]:


#MONTH WISE ANALYSIS
mon=month_data(1)
mon_split=mon['CATEGORY'].value_counts()
mon_split=list(mon_split.index)
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') & (mon['CATEGORY'] == 'WELLNESS') & (mon['CATEGORY'] == 'ENTERTAINMENT') & (mon['CATEGORY'] == 'PARENTING') & (mon['CATEGORY']=='TRAVEL')]
mon_split_1=mon[(mon['CATEGORY'] == 'BUSINESS') & (mon['CATEGORY'] == 'SPORTS') & (mon['CATEGORY'] == 'MEDIA') & (mon['CATEGORY'] == 'ENVIRONMENT') & (mon['CATEGORY']=='EDUCATION')]


# In[121]:


#Jan
mon=month_data(1)
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in JAN \n",x18[:6])
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[ ]:





# In[122]:


#FEB
mon=month_data(2)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in FEB \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[123]:


#MAR
mon=month_data(3)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in MAR \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[124]:


#APR
mon=month_data(4)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in APR \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[125]:


#MAY
mon=month_data(5)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in MAY \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[126]:


#JUN
mon=month_data(6)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in JUN \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[106]:


#JULY
mon=month_data(7)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in JULY \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[127]:


#AUG
mon=month_data(8)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in AUG \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[128]:


#SEP
mon=month_data(9)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in SEP \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[129]:


#OCT
mon=month_data(10)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in OCT \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[130]:


#NOV
mon=month_data(11)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in NOV \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[131]:


#DEC
mon=month_data(12)
x18=mon['AUTHOR'].value_counts()
print("\nTop 5 authors in DEC \n",x18[:6])
mon_split_1=mon[(mon['CATEGORY'] == 'POLITICS') | (mon['CATEGORY'] == 'WELLNESS') | (mon['CATEGORY'] == 'ENTERTAINMENT') | (mon['CATEGORY'] == 'PARENTING') | (mon['CATEGORY']=='TRAVEL')]
mon_split_2=mon[(mon['CATEGORY'] == 'BUSINESS') | (mon['CATEGORY'] == 'SPORTS') | (mon['CATEGORY'] == 'MEDIA') | (mon['CATEGORY'] == 'ENVIRONMENT') | (mon['CATEGORY']=='EDUCATION')]
fig, axes = plt.subplots(2, 1, figsize=(10,10))
sns.despine(left=True)
sns.countplot(x = 'CATEGORY', data = mon_split_1,ax=axes[0])
sns.countplot(x = 'CATEGORY', data = mon_split_2,ax=axes[1])


# In[134]:


mon_data=day_of_week('Monday')
tue_data=day_of_week('Tuesday')
wed_data=day_of_week('Wednesday')
thu_data=day_of_week('Thurday')
fri_data=day_of_week('Friday')
sat_data=day_of_week('Saturday')
sun_data=day_of_week('Sunday')

weekday_data = pd.concat([mon_data,tue_data,wed_data,thu_data])
weekend_data = pd.concat([fri_data,sat_data,sun_data])


# In[145]:


x18=weekday_data['AUTHOR'].value_counts()
print("\nTop 5 authors on Weekdays \n",x18[:6])
print("\nLeast popular 5 authors on Weekdays \n",x18.tail(5))


# In[146]:


x18=weekend_data['AUTHOR'].value_counts()
print("\nTop 5 authors on Weekends \n",x18[:6])
print("\nLeast popular 5 authors on Weekends \n",x18.tail(5))


# In[143]:


x18=weekday_data['CATEGORY'].value_counts()
print("\nTop 5 categories on Weekdays \n",x18[:6])
print("\nLeast 5 categories on Weekdays \n",x18.tail(5))


# In[142]:


x18=weekend_data['CATEGORY'].value_counts()
print("\nTop 5 categories on Weekends \n",x18[:6])
print("\nLeast 5 categories on Weekends \n",x18.tail(5))


# In[ ]:





# In[ ]:





# In[26]:


#AUTHOR PLOT LEE  MORGAN
fig, ax = plt.subplots(1, 1, figsize=(10,10))
df.plot.pie( autopct = '%1.1f%%')


# In[43]:


import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition, ensemble
#from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
x=data_slicer('26-05-2016','29-06-2018')
trainDF = pd.DataFrame()
trainDF['label']=x['CATEGORY']
trainDF['text']=x['HEADLINE']

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# create a count vectorizer object 
'''count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.fit_transform(train_x.values.astype('U'))
xvalid_count =  count_vect.fit_transform(valid_x.values.astype('U'))'''

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print("RF, Count Vectors: ", accuracy)


# In[44]:


from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(xtrain_count,train_y)
Y_predict = model.predict(xvalid_count)
accuracy = accuracy_score(valid_y,Y_predict)*100
print(format(accuracy, '.2f'))


# In[38]:


logistic_Regression = LogisticRegression()
logistic_Regression.fit(xtrain_count,train_y)
Y_predict = logistic_Regression.predict(xvalid_count)
accuracy = accuracy_score(valid_y,Y_predict)*100
print(format(accuracy, '.2f'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




