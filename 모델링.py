# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:24:13 2022

@author: yonghakim
"""

#########모델링

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df_log_tr=pd.read_csv('df_log_tr.csv')
df_log_test=pd.read_csv('df_log_test.csv')

df_log_tr = df_log_tr.replace('EARNEDINCOME',1)
df_log_tr = df_log_tr.replace('EARNEDINCOME2',2)
df_log_tr = df_log_tr.replace('PRIVATEBUSINESS',3)
df_log_tr = df_log_tr.replace('OTHERINCOME',4)
df_log_tr = df_log_tr.replace('FREELANCER',5)
df_log_tr = df_log_tr.replace('PRACTITIONER',6)

df_log_tr = df_log_tr.replace('정규직',1)
df_log_tr = df_log_tr.replace('계약직',2)
df_log_tr['employment_type'] = df_log_tr['employment_type'].replace('기타',3)
df_log_tr = df_log_tr.replace('일용직',4)

df_log_tr = df_log_tr.replace('전월세',1)
df_log_tr = df_log_tr.replace('자가',2)
df_log_tr = df_log_tr.replace('기타가족소유',3)
df_log_tr = df_log_tr.replace('배우자',4)

df_log_tr = df_log_tr.replace('생활비',1)
df_log_tr = df_log_tr.replace('대환대출',2)
df_log_tr = df_log_tr.replace('주택구입',3)
df_log_tr = df_log_tr.replace('전월세보증금',4)
df_log_tr = df_log_tr.replace('사업자금',5)
df_log_tr['purpose'] = df_log_tr['purpose'].replace('기타',6)
df_log_tr = df_log_tr.replace('투자',7)
df_log_tr = df_log_tr.replace('자동차구입',8)

df_log_tr = df_log_tr.astype({'user_id':'category'})
df_log_tr = df_log_tr.astype({'application_id':'category'})
df_log_tr = df_log_tr.astype({'is_applied':'category'})
df_log_tr = df_log_tr.astype({'loan_month':'category'})
df_log_tr = df_log_tr.astype({'gender':'category'})
df_log_tr = df_log_tr.astype({'income_type':'category'})
df_log_tr = df_log_tr.astype({'employment_type':'category'})
df_log_tr = df_log_tr.astype({'houseown_type':'category'})
df_log_tr = df_log_tr.astype({'purpose':'category'})
df_log_tr = df_log_tr.astype({'change':'category'})
df_log_tr = df_log_tr.astype({'credit_scoreC':'category'})
df_log_tr = df_log_tr.astype({'bank_id':'category'})
df_log_tr = df_log_tr.astype({'product_id':'category'})
df_log_tr = df_log_tr.astype({'loan_day':'category'})
df_log_tr = df_log_tr.astype({'birth_year':'category'})
df_log_tr = df_log_tr.astype({'user_month':'category'})
df_log_tr = df_log_tr.astype({'company_enter_m':'category'})
df_log_tr = df_log_tr.astype({'company_enter_y':'category'})
df_log_tr = df_log_tr.astype({'user_day':'category'})
df_log_tr = df_log_tr.astype({'user_weekday':'category'})
df_log_tr = df_log_tr.astype({'loan_weekday':'category'})

df_log_test = df_log_test.replace('EARNEDINCOME',1)
df_log_test = df_log_test.replace('EARNEDINCOME2',2)
df_log_test = df_log_test.replace('PRIVATEBUSINESS',3)
df_log_test = df_log_test.replace('OTHERINCOME',4)
df_log_test = df_log_test.replace('FREELANCER',5)
df_log_test = df_log_test.replace('PRACTITIONER',6)

df_log_test = df_log_test.replace('정규직',1)
df_log_test = df_log_test.replace('계약직',2)
df_log_test['employment_type'] = df_log_test['employment_type'].replace('기타',3)
df_log_test = df_log_test.replace('일용직',4)

df_log_test = df_log_test.replace('전월세',1)
df_log_test = df_log_test.replace('자가',2)
df_log_test = df_log_test.replace('기타가족소유',3)
df_log_test = df_log_test.replace('배우자',4)

df_log_test = df_log_test.replace('생활비',1)
df_log_test = df_log_test.replace('대환대출',2)
df_log_test = df_log_test.replace('주택구입',3)
df_log_test = df_log_test.replace('전월세보증금',4)
df_log_test = df_log_test.replace('사업자금',5)
df_log_test['purpose'] = df_log_test['purpose'].replace('기타',6)
df_log_test = df_log_test.replace('투자',7)
df_log_test = df_log_test.replace('자동차구입',8)

df_log_test = df_log_test.astype({'user_id':'category'})
df_log_test = df_log_test.astype({'application_id':'category'})
df_log_test = df_log_test.astype({'is_applied':'category'})
df_log_test = df_log_test.astype({'loan_month':'category'})
df_log_test = df_log_test.astype({'gender':'category'})
df_log_test = df_log_test.astype({'income_type':'category'})
df_log_test = df_log_test.astype({'employment_type':'category'})
df_log_test = df_log_test.astype({'houseown_type':'category'})
df_log_test = df_log_test.astype({'purpose':'category'})
df_log_test = df_log_test.astype({'change':'category'})
df_log_test = df_log_test.astype({'credit_scoreC':'category'})
df_log_test = df_log_test.astype({'bank_id':'category'})
df_log_test = df_log_test.astype({'product_id':'category'})
df_log_test = df_log_test.astype({'loan_day':'category'})
df_log_test = df_log_test.astype({'birth_year':'category'})
df_log_test = df_log_test.astype({'user_month':'category'})
df_log_test = df_log_test.astype({'company_enter_m':'category'})
df_log_test = df_log_test.astype({'company_enter_y':'category'})
df_log_test = df_log_test.astype({'user_day':'category'})
df_log_test = df_log_test.astype({'user_weekday':'category'})
df_log_test = df_log_test.astype({'loan_weekday':'category'})

df_log_tr1 = df_log_tr.drop(['user_id','user_month','loan_month','month','bank_id'],axis=1)
df_log_test1 = df_log_test.drop(['user_id','user_month','loan_month','month','bank_id'],axis=1)

##purpose=1
df1 = df_log_tr1[df_log_tr1['purpose']==1]
df11 = df_log_test1[df_log_test1['purpose']==1]
X_train = df1.drop(['is_applied','application_id','product_id'], axis=1)
X_test = df11.drop(['is_applied','application_id','product_id'], axis=1)
y_train = df1['is_applied']

rfc = RandomForestClassifier(n_estimators=50, random_state=123)
rfc.fit(X_train, y_train)
y_pred_1 = rfc.predict(X_test)
predict1=df11[['application_id','product_id']]
predict1['is_applied']=y_pred_1
##purpose=2
df2 = df_log_tr1[df_log_tr1['purpose']==2]
df22 = df_log_test1[df_log_test1['purpose']==2]
X_train = df2.drop(['is_applied','application_id','product_id'], axis=1)
X_test = df22.drop(['is_applied','application_id','product_id'], axis=1)
y_train = df2['is_applied']

rfc = RandomForestClassifier(n_estimators=50, random_state=123)
rfc.fit(X_train, y_train)
y_pred_2 = rfc.predict(X_test)
predict2=df22[['application_id','product_id']]
predict2['is_applied']=y_pred_2

df3 = df_log_tr1[df_log_tr1['purpose']==3]
df33 = df_log_test1[df_log_test1['purpose']==3]
X_train = df3.drop(['is_applied','application_id','product_id'], axis=1)
X_test = df33.drop(['is_applied','application_id','product_id'], axis=1)
y_train = df3['is_applied']

rfc = RandomForestClassifier(n_estimators=50, random_state=123)
rfc.fit(X_train, y_train)
y_pred_3 = rfc.predict(X_test)
predict3=df33[['application_id','product_id']]
predict3['is_applied']=y_pred_3

df4 = df_log_tr1[df_log_tr1['purpose']==4]
df44 = df_log_test1[df_log_test1['purpose']==4]
X_train = df4.drop(['is_applied','application_id','product_id'], axis=1)
X_test = df44.drop(['is_applied','application_id','product_id'], axis=1)
y_train = df4['is_applied']

rfc = RandomForestClassifier(n_estimators=50, random_state=123)
rfc.fit(X_train, y_train)
y_pred_4 = rfc.predict(X_test)
predict4=df44[['application_id','product_id']]
predict4['is_applied']=y_pred_4

df5 = df_log_tr1[df_log_tr1['purpose']==5]
df55 = df_log_test1[df_log_test1['purpose']==5]
X_train = df5.drop(['is_applied','application_id','product_id'], axis=1)
X_test = df55.drop(['is_applied','application_id','product_id'], axis=1)
y_train = df5['is_applied']

rfc = RandomForestClassifier(n_estimators=50, random_state=123)
rfc.fit(X_train, y_train)
y_pred_5 = rfc.predict(X_test)
predict5=df55[['application_id','product_id']]
predict5['is_applied']=y_pred_5

df6 = df_log_tr1[df_log_tr1['purpose']==6]
df66 = df_log_test1[df_log_test1['purpose']==6]
X_train = df6.drop(['is_applied','application_id','product_id'], axis=1)
X_test = df66.drop(['is_applied','application_id','product_id'], axis=1)
y_train = df6['is_applied']

rfc = RandomForestClassifier(n_estimators=50, random_state=123)
rfc.fit(X_train, y_train)
y_pred_6 = rfc.predict(X_test)
predict6=df66[['application_id','product_id']]
predict6['is_applied']=y_pred_6

df7 = df_log_tr1[df_log_tr1['purpose']==7]
df77 = df_log_test1[df_log_test1['purpose']==7]
X_train = df7.drop(['is_applied','application_id','product_id'], axis=1)
X_test = df77.drop(['is_applied','application_id','product_id'], axis=1)
y_train = df7['is_applied']

rfc = RandomForestClassifier(n_estimators=50, random_state=123)
rfc.fit(X_train, y_train)
y_pred_7 = rfc.predict(X_test)
predict7=df77[['application_id','product_id']]
predict7['is_applied']=y_pred_7

df8 = df_log_tr1[df_log_tr1['purpose']==8]
df88 = df_log_test1[df_log_test1['purpose']==8]
X_train = df8.drop(['is_applied','application_id','product_id'], axis=1)
X_test = df88.drop(['is_applied','application_id','product_id'], axis=1)
y_train = df8['is_applied']

rfc = RandomForestClassifier(n_estimators=50, random_state=123)
rfc.fit(X_train, y_train)

y_pred_8 = rfc.predict(X_test)
predict8=df88[['application_id','product_id']]
predict8['is_applied']=y_pred_8

predict= pd.concat([predict1, predict2,predict3,predict4,predict5,predict6,predict7,predict8], axis = 0)

predict=predict.drop_duplicates()

p0=predict[predict['is_applied']==0]
p1=predict[predict['is_applied']==1]
predict0=pd.merge(p0,p1,how='inner',left_on=['application_id','product_id'],right_on=['application_id','product_id'])

predict[(predict['application_id']==1023905)&(predict['product_id']==216)&(predict['is_applied']==0)]
predict[(predict['application_id']==1372074)&(predict['product_id']==108)&(predict['is_applied']==0)]
predict[(predict['application_id']==1372074)&(predict['product_id']==236)&(predict['is_applied']==0)]
predict[(predict['application_id']==1425702)&(predict['product_id']==226)&(predict['is_applied']==0)]

bug=[2139813,2061029,2061025,1533503]

predict.drop(bug, inplace=True)

test=pd.read_csv('데이터분석분야_퓨처스부문_평가데이터.csv')

predict_fin=pd.merge(test,predict,how='left',left_on=['application_id','product_id'],right_on=['application_id','product_id'])

predict_fin.drop(['is_applied_x'],axis=1,inplace=True)

predict_fin.rename(columns={'is_applied_y':'is_applied'},inplace=True)

predict_fin.to_csv('빅콘테스트 데이터분석분야 퓨처스 부문 평가데이터.csv',index=False)

