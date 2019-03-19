#IMPORT LIBS 
import pandas as pd
import numpy as np
import datetime

#IMPORT DATA
master_data=pd.read_csv("master_data.csv")

#CLEAN DATA
master_data['AUTHOR'] = master_data['AUTHOR'].fillna('None')



#DATA FUNCTIONS

def data_slicer(start_date,end_date):
	index = pd.date_range(start=start_date,end=end_date)
	date_list=[]
	for ch in index:
		datee = datetime.datetime.strptime(str(ch), "%Y-%m-%d %H:%M:%S")
		datee2=datee.strftime("%d-%m-%Y")
		date_list.append(datee2)
	data_req=master_data[master_data['DATE'].isin(date_list)]
	return data_req

#print(data_slicer('26-05-2016','29-06-2016'))

def month_data(month_number):
	data_req=master_data[master_data['MONTH'] == month_number]
	return data_req

#print(month_data(6))

def day_of_week(day_name):
	data_req=master_data[master_data['DAY_OF_WEEK'] == day_name]
	return data_req

#print(day_of_week('Monday'))

def data_author(author_name):
	data_req=master_data[master_data.AUTHOR.str.contains(author_name)]
	return data_req

#print(data_author('Ron'))

#ANALYSIS FUNCTIONS

def date_to_date_analysis(start_date,end_date):
	data=data_slicer(start_date,end_date)
	category_count={}
	author_article_count={}

	year_list=unique(list(data['YEAR']))
	category_list=unique(list(data['CATEGORY']))
	authors_list=unique(list(data['AUTHOR']))
	print("The active year/s in the given range :",year_list)
	print("The number of categories published :",len(category_list))
	print("The categories published :",category_list)
	print("The number of active authors :",len(authors_list))
	print("The active authors in this date range :",authors_list)

	for cat in category_list:
		data1=data[data['CATEGORY']==cat]
		category_count[cat] = data1['CATEGORY'].count()
	print(category_count)

	for cat in authors_list:
		data1=data[data['AUTHOR']==cat]
		author_article_count[cat] = data1['AUTHOR'].count()
	print(author_article_count)

	high_auth=[]
	highest_count=0
	highest_name=None
	for name,count in author_article_count:
		if count>highest_count:
			highest_count=count
			highest_name=name
		if count > 10:
			high_auth.append(name)
	
	#print("\nAuthors who wrote more than 10 posts:",high_auth)
	print("\nMost Active Author in the given date range :",highest_name+' '+highest_count)




