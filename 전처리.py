# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 20:29:01 2022

@author: yonghakim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

##user 전처리
user=pd.read_csv('user_spec.csv')


##purpose 일치시키기
user=user.replace('LIVING','생활비')
user=user.replace('SWITCHLOAN','대환대출')
user=user.replace('BUSINESS','사업자금')
user=user.replace('ETC','기타')
user=user.replace('HOUSEDEPOSIT','전월세보증금')
user=user.replace('BUYHOUSE','주택구입')
user=user.replace('INVEST','투자')
user=user.replace('BUYCAR','자동차구입')

user['insert_time'] = pd.to_datetime(user['insert_time'])
user['user_month']=user['insert_time'].dt.month

##6월과 나머지 분할
usertr=user[user['user_month']!=6]
usertest=user[user['user_month']==6]



##3,4,5월 데이터 입사년월 분할

usertr['company_enter_month'] = usertr['company_enter_month'].fillna(0)

usertr['company_enter_month']=usertr['company_enter_month'].astype(int)


enter=divmod(usertr['company_enter_month'], 100)

usertr['company_enter_m']= enter[1]
usertr['company_enter_y']= enter[0]
##6월 데이터 입사년월 분할
usertest['company_enter_month'] = usertest['company_enter_month'].fillna(0)

usertest['company_enter_month'] = usertest['company_enter_month'].astype(int)

com1=divmod(usertest['company_enter_month'], 100)

com2=divmod(com1[0], 100)

user['company_enter_m']=pd.concat([enter[0],com2[0]], axis = 0)
user['company_enter_y']=pd.concat([enter[1],com2[1]], axis = 0)

##6월과 나머지 분할
usertr=user[user['user_month']!=6]
usertest=user[user['user_month']==6]


##change 변수 만들기
usery = user.groupby("user_id")['yearly_income'].nunique()
useri = user.groupby("user_id")['income_type'].nunique()
userc = user.groupby("user_id")['company_enter_month'].nunique()
usere = user.groupby("user_id")['employment_type'].nunique()
userh = user.groupby("user_id")['houseown_type'].nunique()

usery = usery.to_frame().reset_index()
useri = useri.to_frame().reset_index()
userc = userc.to_frame().reset_index()
usere = usere.to_frame().reset_index()
userh = userh.to_frame().reset_index()

usery= usery[usery['yearly_income']>1]
useri= useri[useri['income_type']>1]
userc= userc[userc['company_enter_month']>1]
usere= usere[usere['employment_type']>1]
userh= userh[userh['houseown_type']>1]


user1= pd.concat([usery, useri,userc,usere,userh], axis = 0)
user_idu=user1['user_id'].unique()

column_names= ['user_id']
userid = pd.DataFrame(user_idu, columns=column_names)
userid['change']=1

user = pd.merge(user,userid, how='left', left_on='user_id', right_on='user_id')
user['change'] = user['change'].fillna(0)


##생년월일 결측치 처리(겹치는 user_id에 한해)
userbirth=user[user['birth_year'].isnull()]
userbirthn=user[user['birth_year'].notnull()]
userinner=pd.merge(userbirthn,userbirth,how='inner',left_on='user_id',right_on='user_id')

userinner['birth_year_y']=userinner['birth_year_x'] 
userinner['gender_y']=userinner['gender_x'] 

userbir=userinner.loc[:,['application_id_y', 'user_id', 'birth_year_y', 'gender_y', 'insert_time_y','credit_score_y', 'yearly_income_y', 'income_type_y', 'company_enter_month_y','employment_type_y', 'houseown_type_y', 'desired_amount_y', 'purpose_y','personal_rehabilitation_yn_y','personal_rehabilitation_complete_yn_y','existing_loan_cnt_y', 'existing_loan_amt_y', 'user_month_y', 'company_enter_m_y','company_enter_y_y', 'change_y']]
userbiru=userbir.drop_duplicates()

userbiru.columns =['application_id', 'user_id', 'birth_year', 'gender', 'insert_time','credit_score', 'yearly_income', 'income_type', 'company_enter_month','employment_type', 'houseown_type', 'desired_amount', 'purpose','personal_rehabilitation_yn','personal_rehabilitation_complete_yn','existing_loan_cnt', 'existing_loan_amt', 'user_month', 'company_enter_m','company_enter_y', 'change']

a=pd.merge(userbirth,userbiru,how='left',left_on='application_id',right_on='application_id')

a['birth_year_x']=a['birth_year_y'] 
a['gender_x']=a['gender_y']

aa=a.iloc[:,0:21]
aa.columns =['application_id', 'user_id', 'birth_year', 'gender', 'insert_time','credit_score', 'yearly_income', 'income_type', 'company_enter_month','employment_type', 'houseown_type', 'desired_amount', 'purpose','personal_rehabilitation_yn','personal_rehabilitation_complete_yn','existing_loan_cnt', 'existing_loan_amt', 'user_month', 'company_enter_m','company_enter_y', 'change']

user=pd.concat([aa,userbirthn],axis=0)
################
##credit_score
usercre=user[user['credit_score'].isnull()]
usercren=user[user['credit_score'].notnull()]
userinner2=pd.merge(usercren,usercre,how='inner',left_on='user_id',right_on='user_id')

userinner2['credit_score_y']=userinner2['credit_score_x'] 

usercredit=userinner2.loc[:,['application_id_y', 'user_id', 'birth_year_y', 'gender_y', 'insert_time_y','credit_score_y', 'yearly_income_y', 'income_type_y', 'company_enter_month_y','employment_type_y', 'houseown_type_y', 'desired_amount_y', 'purpose_y','personal_rehabilitation_yn_y','personal_rehabilitation_complete_yn_y','existing_loan_cnt_y', 'existing_loan_amt_y', 'user_month_y', 'company_enter_m_y','company_enter_y_y', 'change_y']]
usercreditu=usercredit.drop_duplicates()

usercreditu.columns =['application_id', 'user_id', 'birth_year', 'gender', 'insert_time','credit_score', 'yearly_income', 'income_type', 'company_enter_month','employment_type', 'houseown_type', 'desired_amount', 'purpose','personal_rehabilitation_yn','personal_rehabilitation_complete_yn','existing_loan_cnt', 'existing_loan_amt', 'user_month', 'company_enter_m','company_enter_y', 'change']

b=pd.merge(usercre,usercreditu,how='left',left_on='application_id',right_on='application_id')

b['credit_score_x']=b['credit_score_y']

bb=b.iloc[:,0:21]
bb.columns =['application_id', 'user_id', 'birth_year', 'gender', 'insert_time','credit_score', 'yearly_income', 'income_type', 'company_enter_month','employment_type', 'houseown_type', 'desired_amount', 'purpose','personal_rehabilitation_yn','personal_rehabilitation_complete_yn','existing_loan_cnt', 'existing_loan_amt', 'user_month', 'company_enter_m','company_enter_y', 'change']

user=pd.concat([bb,usercren],axis=0)

###existing_loan_cnt 0으로 대체,existing_loan_amt 0=0,1=해당 그룹 평균으로 대체

usertr=user[user['user_month']!=6]
usertest=user[user['user_month']==6]
usertr['existing_loan_cnt'] = usertr['existing_loan_cnt'].fillna(0)

userex1=usertr[usertr['existing_loan_amt'].isnull()]
userexn1=usertr[usertr['existing_loan_amt'].notnull()]
amtnull=usertr.groupby(['existing_loan_cnt','purpose','income_type'], as_index=False)['existing_loan_amt'].mean()

amtnull=amtnull[amtnull['existing_loan_cnt']==1]

k=pd.merge(userex1,amtnull,how='left',left_on=['existing_loan_cnt','purpose','income_type'],right_on=['existing_loan_cnt','purpose','income_type'])

k['existing_loan_amt_x']=k['existing_loan_amt_y']
k.rename(columns={'existing_loan_amt_x':'existing_loan_amt'}, inplace=True)
k.drop('existing_loan_amt_y',axis=1,inplace=True)
k['existing_loan_amt'] = k['existing_loan_amt'].fillna(0)

usertr=pd.concat([k,userexn1],axis=0)
##test

usertest['existing_loan_cnt'] = usertest['existing_loan_cnt'].fillna(0)
userex11=usertest[usertest['existing_loan_amt'].isnull()]
userexn11=usertest[usertest['existing_loan_amt'].notnull()]
amtnull1=usertr.groupby(['existing_loan_cnt','purpose','income_type'], as_index=False)['existing_loan_amt'].mean()

amtnull1=amtnull1[amtnull1['existing_loan_cnt']==1]

j=pd.merge(userex11,amtnull1,how='left',left_on=['existing_loan_cnt','purpose','income_type'],right_on=['existing_loan_cnt','purpose','income_type'])

j['existing_loan_amt_x']=j['existing_loan_amt_y']
j.rename(columns={'existing_loan_amt_x':'existing_loan_amt'}, inplace=True)
j.drop('existing_loan_amt_y',axis=1,inplace=True)
j['existing_loan_amt'] = j['existing_loan_amt'].fillna(0)

usertest=pd.concat([j,userexn11],axis=0)
##생년월일 결측치 제거

a = round(usertr[usertr['employment_type']=='정규직']['gender'].mean())
b = round(usertr[usertr['employment_type']=='기타']['gender'].mean())
c = round(usertr[usertr['employment_type']=='계약직']['gender'].mean())
d = round(usertr[usertr['employment_type']=='일용직']['gender'].mean())

a2 = round(usertr[usertr['employment_type']=='정규직']['birth_year'].mean())
b2 = round(usertr[usertr['employment_type']=='기타']['birth_year'].mean())
c2 = round(usertr[usertr['employment_type']=='계약직']['birth_year'].mean())
d2 = round(usertr[usertr['employment_type']=='일용직']['birth_year'].mean())

usertr.loc[usertr['gender'].isna() & (usertr['employment_type']=='정규직'),'gender'] = a
usertr.loc[usertr['gender'].isna() & (usertr['employment_type']=='기타'),'gender'] = b
usertr.loc[usertr['gender'].isna() & (usertr['employment_type']=='계약직'),'gender'] = c
usertr.loc[usertr['gender'].isna() & (usertr['employment_type']=='일용직'),'gender'] = d

usertr.loc[usertr['birth_year'].isna() & (usertr['employment_type']=='정규직'),'birth_year'] = a2
usertr.loc[usertr['birth_year'].isna() & (usertr['employment_type']=='기타'),'birth_year'] = b2
usertr.loc[usertr['birth_year'].isna() & (usertr['employment_type']=='계약직'),'birth_year'] = c2
usertr.loc[usertr['birth_year'].isna() & (usertr['employment_type']=='일용직'),'birth_year'] = d2



usertest.loc[usertest['gender'].isna() & (usertest['employment_type']=='정규직'),'gender'] = a
usertest.loc[usertest['gender'].isna() & (usertest['employment_type']=='기타'),'gender'] = b
usertest.loc[usertest['gender'].isna() & (usertest['employment_type']=='계약직'),'gender'] = c
usertest.loc[usertest['gender'].isna() & (usertest['employment_type']=='일용직'),'gender'] = d

usertest.loc[usertest['birth_year'].isna() & (usertest['employment_type']=='정규직'),'birth_year'] = a2
usertest.loc[usertest['birth_year'].isna() & (usertest['employment_type']=='기타'),'birth_year'] = b2
usertest.loc[usertest['birth_year'].isna() & (usertest['employment_type']=='계약직'),'birth_year'] = c2
usertest.loc[usertest['birth_year'].isna() & (usertest['employment_type']=='일용직'),'birth_year'] = d2

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']>=54481460),'credit_score']=1000

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<54481460)&(usertr['yearly_income']>=43452344),'credit_score']=942

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<43452344)&(usertr['yearly_income']>=39303951),'credit_score']=891

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<39303951)&(usertr['yearly_income']>=36776016),'credit_score']=832

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<36776016)&(usertr['yearly_income']>=34797303),'credit_score']=768

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<34797303)&(usertr['yearly_income']>=34424775),'credit_score']=698

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<34424775)&(usertr['yearly_income']>=33627894),'credit_score']=630

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<33627894)&(usertr['yearly_income']>=33267382),'credit_score']=530

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<33267382)&(usertr['yearly_income']>=31883412),'credit_score']=454

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<31883412)&(usertr['yearly_income']>=30091698),'credit_score']=335

usertr.loc[usertr['credit_score'].isna() & (usertr['yearly_income']<30091698),'credit_score']=167

##usertest credit score 대체
usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']>=54481460),'credit_score']=1000

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<54481460)&(usertest['yearly_income']>=43452344),'credit_score']=942

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<43452344)&(usertest['yearly_income']>=39303951),'credit_score']=891

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<39303951)&(usertest['yearly_income']>=36776016),'credit_score']=832

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<36776016)&(usertest['yearly_income']>=34797303),'credit_score']=768

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<34797303)&(usertest['yearly_income']>=34424775),'credit_score']=698

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<34424775)&(usertest['yearly_income']>=33627894),'credit_score']=630

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<33627894)&(usertest['yearly_income']>=33267382),'credit_score']=530

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<33267382)&(usertest['yearly_income']>=31883412),'credit_score']=454

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<31883412)&(usertest['yearly_income']>=30091698),'credit_score']=335

usertest.loc[usertest['credit_score'].isna() & (usertest['yearly_income']<30091698),'credit_score']=167




usertr.drop(['company_enter_month','personal_rehabilitation_yn','personal_rehabilitation_complete_yn'],axis=1,inplace=True)
usertest.drop(['company_enter_month','personal_rehabilitation_yn','personal_rehabilitation_complete_yn'],axis=1,inplace=True)



user=pd.concat([usertr,usertest],axis=0)

user.loc[user['credit_score']>=891,'credit_scoreC']='1'
user.loc[(user['credit_score']>=630) & (user['credit_score']<=890),'credit_scoreC']='2'
user.loc[(user['credit_score']>=454) & (user['credit_score']<=629),'credit_scoreC']='3'
user.loc[user['credit_score'] <= 453,'credit_scoreC']='4'

user['yearly_income1'] = user['yearly_income']/1000000
user['yearly_incomeC'] = 0
user.loc[user['yearly_income1']<=12,'yearly_incomeC']='1'
user.loc[(user['yearly_income1']>12) & (user['yearly_income1']<=46),'yearly_incomeC']='2'
user.loc[(user['yearly_income1']>46) & (user['yearly_income1']<=88),'yearly_incomeC']='3'
user.loc[(user['yearly_income1']>88) & (user['yearly_income1']<=150),'yearly_incomeC']='4'
user.loc[user['yearly_income1']>150,'yearly_incomeC']='5'

user['user_day']=user['insert_time'].dt.day
user['user_weekday']=user['insert_time'].dt.weekday
user.drop(['insert_time','yearly_income1'],axis=1,inplace=True)




##loan 데이터

loan=pd.read_csv('loan_result.csv')
loan=loan[loan['loan_limit'].notnull()]
loan['loanapply_insert_time'] = pd.to_datetime(loan['loanapply_insert_time'])
loan['loan_month']=loan['loanapply_insert_time'].dt.month
loan['loan_day']=loan['loanapply_insert_time'].dt.day
loan['loan_weekday']=loan['loanapply_insert_time'].dt.weekday
loan.drop(['loanapply_insert_time'],axis=1,inplace=True)

##6월과 나머지 분할
loantr=loan[loan['loan_month']!=6]
loante=loan[loan['loan_month']==6]
test=pd.read_csv('데이터분석분야_퓨처스부문_평가데이터.csv')
loantest=pd.merge(test,loan,how='left',left_on=['application_id','product_id'],right_on=['application_id','product_id'])
loantest.drop(['is_applied_x'],axis=1,inplace=True)
loantest.rename(columns={'is_applied_y':'is_applied'},inplace=True)

ultr=pd.merge(loantr,user,how='inner',left_on='application_id',right_on='application_id')
ultest=pd.merge(loantest,user,how='inner',left_on='application_id',right_on='application_id')
ultest.isna().sum()
ultest['yearly_income']=ultest['yearly_income'].fillna(0)


##log
log = pd.read_csv('log_data.csv')
log = log.drop(['mp_os','mp_app_version'], axis=1)
log['date_cd'] = pd.to_datetime(log['date_cd'])
log_dummy = pd.get_dummies(log, columns = ['event'])
log_dummy["month"] = log_dummy["date_cd"].dt.month
log_dummy2 = log_dummy.groupby(['user_id','month']).sum()[['event_EndLoanApply','event_UseLoanManage','event_UsePrepayCalc',
                                                          'event_UseDSRCalc','event_GetCreditInfo']]
log_dummy2.reset_index(inplace=True)
log_pivot = log_dummy2.pivot_table(index='user_id', columns='month', 
                       values=['event_EndLoanApply','event_UseLoanManage','event_UsePrepayCalc','event_UseDSRCalc','event_GetCreditInfo'], aggfunc='sum')
log_pivot2 = log_pivot.copy()
log_pivot2.columns = log_pivot.columns.values

log_pivot2.reset_index(level=0, inplace=True)
ul = pd.read_csv('user_spec.csv')


ul2 = ul[['user_id','insert_time']]
ul2['insert_time'] = pd.to_datetime(ul2['insert_time']).dt.date # 연월일
ul2['insert_time'] = pd.to_datetime(ul2['insert_time']) 

ul2['month'] = ul2['insert_time'].dt.month

df = pd.merge(ul2, log_pivot2, left_on = 'user_id', right_on = 'user_id', how='left' )

ul2_march = ul2[ul2['month']==3]
print('3월 행의 개수 : ',len(ul2_march))
print('\n')

ul2_april = ul2[ul2['month']==4]
print('4월 행의 개수 : ',len(ul2_april))
print('\n')

ul2_may = ul2[ul2['month']==5]
print('5월 행의 개수 : ',len(ul2_may))
print('\n')

ul2_june = ul2[ul2['month']==6]
print('6월 행의 개수 : ',len(ul2_june))
print('\n')

print('행의 개수 합 확인 : ',len(ul2_march) + len(ul2_april) + len(ul2_may) + len(ul2_june), '=' ,len(ul2))

march = pd.merge(ul2_march, log_pivot2[['user_id']], left_on = 'user_id', right_on = 'user_id', how='left' )

march['EndLoanApply_beforemonth'] = 0
march['UseLoanManage_beforemonth'] = 0
march['UsePrepayCalc_beforemonth'] = 0
march['UseDSRCalc_beforemonth'] = 0
march['GetCreditinfo_beforemonth'] = 0

march.fillna(0)

march

april = pd.merge(ul2_april, log_pivot2[['user_id',
                                        ('event_EndLoanApply', 3), ('event_UseLoanManage', 3),('event_UsePrepayCalc', 3),('event_UseDSRCalc', 3),('event_GetCreditInfo', 3) ]], left_on = 'user_id', right_on = 'user_id', how='left' )


april = april.rename(columns={('event_EndLoanApply', 3):'EndLoanApply_beforemonth',
                             ('event_UseLoanManage', 3):'UseLoanManage_beforemonth',
                             ('event_UsePrepayCalc', 3):'UsePrepayCalc_beforemonth',
                             ('event_UseDSRCalc', 3):'UseDSRCalc_beforemonth',
                             ('event_GetCreditInfo', 3):'GetCreditinfo_beforemonth'})

april.iloc[:,3:] = april.iloc[:,3:].fillna(0)

april

may = pd.merge(ul2_may, log_pivot2[['user_id',
                                        ('event_EndLoanApply', 4), ('event_UseLoanManage', 4),('event_UsePrepayCalc', 4),('event_UseDSRCalc', 4),('event_GetCreditInfo', 4) ]], left_on = 'user_id', right_on = 'user_id', how='left' )


may = may.rename(columns={('event_EndLoanApply', 4):'EndLoanApply_beforemonth',
                             ('event_UseLoanManage', 4):'UseLoanManage_beforemonth',
                             ('event_UsePrepayCalc', 4):'UsePrepayCalc_beforemonth',
                             ('event_UseDSRCalc', 4):'UseDSRCalc_beforemonth',
                             ('event_GetCreditInfo', 4):'GetCreditinfo_beforemonth'})

may.iloc[:,3:] = may.iloc[:,3:].fillna(0)

may

june = pd.merge(ul2_june, log_pivot2[['user_id',
                                        ('event_EndLoanApply', 5),('event_UseLoanManage', 5),('event_UsePrepayCalc', 5),('event_UseDSRCalc', 5),('event_GetCreditInfo', 5) ]], left_on = 'user_id', right_on = 'user_id', how='left' )
june = june.rename(columns={('event_EndLoanApply', 5):'EndLoanApply_beforemonth',
                             ('event_UseLoanManage', 5):'UseLoanManage_beforemonth',
                             ('event_UsePrepayCalc', 5):'UsePrepayCalc_beforemonth',
                             ('event_UseDSRCalc', 5):'UseDSRCalc_beforemonth',
                             ('event_GetCreditInfo', 5):'GetCreditinfo_beforemonth'})

june.iloc[:,3:] = june.iloc[:,3:].fillna(0)

june

df_list = [march, april, may, june]
log_fin = pd.concat(df_list, ignore_index=True)
log_fin=log_fin.drop(['insert_time'],axis=1)
log_fin=log_fin.drop_duplicates()



df_log_tr=pd.merge(ultr,log_fin,how='left',left_on=['user_id','loan_month'],right_on=['user_id','month'])
df_log_test=pd.merge(ultest,log_fin,how='left',left_on=['user_id','loan_month'],right_on=['user_id','month'])


df_log_tr.to_csv('df_log_tr.csv',index=False)
df_log_tr.to_csv('ultr.csv',index=False)
df_log_test.to_csv('df_log_test.csv',index=False)
















