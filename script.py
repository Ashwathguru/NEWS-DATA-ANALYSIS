from pandas.io.json import json_normalize
import json
import pandas as pd
import numpy as np
import datetime
from datetime import date
import calendar

data = pd.read_json('News_Category_Dataset_v2.json', lines=True)
#print(data.columns.values)

idx=list(np.arange(1,200854))
date_list=list(data['date'])
author_list=list(data['authors'])
category_list=list(data['category'])
headline_list=list(data['headline'])
link_list=list(data['link'])
short_list=list(data['short_description'])

#print(date_list)
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

#print(weekday)



clean_data=pd.DataFrame({'SERIAL_NUMBER':idx,'DATE':date_list,'YEAR':year_list,'MONTH':month_list,'DAY':day,'DAY_OF_WEEK':weekday,'AUTHOR':author_list,'CATEGORY':category_list,'HEADLINE':headline_list,'LINK':link_list,'SHORT_DESC':short_list})

#print(clean_data)

clean_data.to_csv('master_data.csv',sep=',',header=True)