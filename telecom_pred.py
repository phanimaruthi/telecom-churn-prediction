# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,recall_score,precision_score,f1_score,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.ensemble import BaggingClassifier

# %%
churn_df=pd.read_csv(r'C:\Users\MARUTHI\Downloads\telecom_pred\churn.csv')
churn_df

# %%
churn_df.info()

# %%
churn_df.isna().sum()

# %%
column=list(churn_df.columns)
for i in range(len(column)):
    column[i]=column[i].lower()
churn_df.columns=column


# %%
churn_df['totalcharges'].astype('float64')

# %%
churn_df[churn_df['totalcharges']==' ']

# %%
churn_df['gender'].value_counts()

# %%
churn_df.columns

# %%
mask=churn_df['totalcharges']==' '
churn_df.loc[mask,'totalcharges']=churn_df.loc[mask,'tenure']*churn_df.loc[mask,'monthlycharges']
churn_df.iloc[936]

# %%
churn_df['totalcharges']=churn_df['totalcharges'].astype('float64')

# %%
churn_df.info()

# %%
sns.countplot(x='gender',data=churn_df,hue='churn')
plt.show()

# %%
churn_df['churn'].value_counts()

# %%
churn_df['gender'].value_counts()  #output=gender is not considered as the X variable since it is unable to much differentiate for target.

# %%
grouped=churn_df.groupby('seniorcitizen')['churn'].value_counts().reset_index()

# %%
grouped

# %%
grouped['sen_churn']=grouped['seniorcitizen'].astype(str)+'_'+grouped['churn']

# %%
grouped

# %%
plt.pie(grouped['count'],data=grouped,labels=grouped['sen_churn'])
plt.show()

# %%
churn_df['seniorcitizen'].value_counts() #not much but can be considered since it is having more ratio to seperate if he is not 
                                         #senior citizen               

# %%
depend=churn_df.groupby('dependents')['churn'].value_counts().reset_index()

# %%
depend

# %%
depend['dep_churn']=depend['dependents']+'_'+depend['churn']
depend

# %%
sns.barplot(x='dep_churn',y='count',data=depend)
plt.show()

# %%
churn_df['dependents'].value_counts()  #output=cannot much differntiate so cannot consider this.

# %%
churn_df['tenure'].max()

# %%
churn_df['tenure'].min()

# %%
churn_df['tenure'].mean()

# %%
churn_df[churn_df['tenure']<=32]['churn'].value_counts()

# %%
churn_df['churn'].value_counts()

# %%
churn_df[churn_df['tenure']<=30]['churn'].value_counts()

# %%
churn_df[(churn_df['tenure']>32)]['churn'].value_counts()

# %%
sns.histplot(data=churn_df, x='tenure', hue='churn',multiple='dodge')
plt.show()

# %%
#output=we could easily get the target variable yes or no since most of the customers are churning in a short period of time
        #and when tenure is more churning amount is less.

# %%
churn_df.head()

# %%
churn_df['phoneservice'].value_counts()

# %%
table=pd.crosstab(churn_df['phoneservice'],churn_df['churn'])
sns.heatmap(table,annot=True)
plt.show()

# %%
churn_df['phoneservice'].value_counts()

# %%
churn_df['multiplelines'].value_counts()

# %%
table=pd.crosstab(churn_df['multiplelines'],churn_df['churn'])

# %%
table

# %%
sns.heatmap(table,annot=True)
plt.show()

# %%
#output=let us check some more

# %%
churn_df.columns

# %%
sns.swarmplot(y='tenure',x='multiplelines',data=churn_df,hue='churn')
plt.show()

# %%
#can be considered at some extent but not totally

# %%
churn_df['internetservice'].value_counts()

# %%
table=pd.crosstab(churn_df['internetservice'],churn_df['churn'])
table

# %%
#output =by seeing this most of the people who selected fiberoptic had gone for churning
#so this can be considered and let us go for some more investigation

# %%
sns.stripplot(x='internetservice',y='tenure',data=churn_df,hue='churn')
plt.show()

# %%
sns.swarmplot(x='internetservice',y='tenure',data=churn_df,hue='churn')
plt.show()

# %%
#output = could understand that mostly who chose for fiberoptic are churning and it can be considered as a factor

# %%
churn_df.columns

# %%
churn_df['onlinesecurity'].value_counts()

# %%
sns.countplot(x='onlinesecurity',data=churn_df,hue='churn')
plt.show()

# %%
table=pd.crosstab(churn_df['onlinesecurity'],churn_df['churn'])
table

# %%
sns.swarmplot(x='onlinesecurity',y='tenure',data=churn_df,hue='churn')
plt.show()

# %%
#output=can be considered 

# %%
churn_df['onlinebackup'].value_counts()

# %%
table=pd.crosstab(churn_df['onlinebackup'],churn_df['churn'])
sns.heatmap(table,annot=True)
plt.show()

# %%
sns.countplot(x='onlinebackup',data=churn_df,hue='churn')
plt.show()

# %%
#output=can be considered at some extent as onlinebackup is dependent on internetservice and internetservice is considered so mostly it 
        #is not considered

# %%
churn_df['deviceprotection'].value_counts()

# %%
#output=Since device protection is mostly dependent on internetservice so it is not mostly considered as the dependent one
        #being considered that is internetservice

# %%
#As deviceprotection and internetservice has almost same value counts  and has most same values it is being dependent on internetservice

# %%
churn_df['techsupport'].value_counts()

# %%
#output = Since techsupport is mostly dependent on online security so it is not mostly considered as the dependent one
          # is being considered that is onlinesecurity

# %%
#As onlinesecurity and techsupport has almost same value counts  and has most same values it is being dependent on onlinesecurity

# %%
churn_df.columns

# %%
churn_df['streamingtv'].value_counts()

# %%
grouped=churn_df.groupby('streamingtv')['churn'].value_counts().reset_index()

# %%
grouped

# %%
churned=grouped[grouped['churn']=='Yes']
nochurn=grouped[grouped['churn']=='No']
print(nochurn)
churned

# %%
plt.pie(churned['count'],data=churned,labels=churned['streamingtv'])
plt.title('churning')
plt.show()

# %%
plt.pie(nochurn['count'],data=nochurn,labels=nochurn['streamingtv'])
plt.title('nochurn')
plt.show()

# %%
#output = in churning streamingtv and not streamingtv contributing same amount and cannot be easily differentiated
          #so it cannot be considered

# %%
churn_df['contract'].value_counts()

# %%
table=pd.crosstab(churn_df['contract'],churn_df['churn'])
table

# %%
#it can be considered since mostly month-to-month contract are churning mostly

# %%
churn_df['paperlessbilling'].value_counts()

# %%
sns.countplot(x='paperlessbilling',data=churn_df,hue='churn')
plt.show()

# %%
churn_df['churn'].value_counts()

# %%
#let us investigate some more

# %%
sns.stripplot(x='paperlessbilling',y='tenure',data=churn_df,hue='churn')
plt.show()

# %%
#output = can be considered at some extent as it is able to get and differentiate churned but unable to differentiate not churned ones to
          #at some extent as it has minimally same value counts

# %%
churn_df.columns

# %%
churn_df['paymentmethod'].value_counts()

# %%
table=pd.crosstab(churn_df['paymentmethod'],churn_df['churn'])
sns.heatmap(table,annot=True)
plt.show()

# %%
#output = cannot be considered since most of the values are same

# %%
churn_df['monthlycharges'].dtype

# %%
churn_df['monthlycharges'].mean()

# %%
churn_df['monthlycharges'].max(),churn_df['monthlycharges'].min()

# %%
churn_df[churn_df['monthlycharges']<=62.76]['churn'].value_counts()

# %%
churn_df[churn_df['monthlycharges']>62.76]['churn'].value_counts()

# %%
#should investigate some more

# %%
sns.scatterplot(x='tenure',y='monthlycharges',data=churn_df,hue='churn')
plt.show()

# %%
sns.swarmplot(x='internetservice',y='tenure',data=churn_df,hue='churn')
plt.show()

# %%
sns.swarmplot(x='internetservice',y='monthlycharges',data=churn_df,hue='churn')
plt.show()

# %%
#output = people who opted for fibre optics have more monthly charges and are churning and monthly charges are dependent on 
         #internetservice and can be considered as when plotted against tenure is able differentiate mostly.

# %%
churn_df['totalcharges'].min(),churn_df['totalcharges'].max()

# %%
churn_df['totalcharges'].mean()

# %%
churn_df[churn_df['totalcharges']>2279.734]['churn'].value_counts()

# %%
churn_df[churn_df['totalcharges']<=2279.734]['churn'].value_counts()

# %%
sns.stripplot(x='internetservice',y='totalcharges',data=churn_df,hue='churn')
plt.show()

# %%
plt.figure(figsize=[10,8])
sns.scatterplot(x='tenure',y='totalcharges',hue='churn',style='internetservice',data=churn_df)
plt.show()

# %%
#output=totalcharges can be considered as a dependent variable

# %%
#let us check if two variables on a combination can predict churns at a higher rate

# %%
churn_df['internetservice'] = churn_df['internetservice'].apply(lambda x: x.lstrip().lower() if isinstance(x, str) else x)

# %%
churn_df['internetservice'].value_counts()

# %%
churn_df['internetservice'].value_counts()

# %%
dsl=churn_df[churn_df['internetservice']=='dsl']

# %%
dsl

# %%
dsl.monthlycharges.max()

# %%
churn_df[churn_df['internetservice']=='fiber optic']['monthlycharges'].mean()

# %%
dsl.monthlycharges.min()

# %%
dsl.tenure.mean()

# %%
tenu_32=dsl[dsl['tenure']<=32.82]

# %%
dsl.churn.value_counts()

# %%
tenu_32

# %%
tenu_32.monthlycharges.mean()

# %%
dsl.monthlycharges.mean()

# %%
tenu_32.groupby('onlinebackup')['churn'].value_counts()

# %%
tenu_32.columns

# %%
tenu_32.groupby('multiplelines')['churn'].value_counts()

# %%
tenu_32.shape

# %%
tenu_32.groupby('onlinesecurity')['churn'].value_counts()

# %%
churn_df['internetservice'].value_counts()

# %%
tenu_32.columns

# %%
tenu_32.groupby('deviceprotection')['churn'].value_counts()

# %%
churned=tenu_32[tenu_32['churn']=='Yes']

# %%
churned

# %%
churned['multiplelines'].value_counts()

# %%
churned['tenure'].max()

# %%
churned['paperlessbilling'].value_counts()

# %%
churned['deviceprotection'].value_counts()

# %%
churn_df['deviceprotection'].value_counts()

# %%
churned['phoneservice'].value_counts()

# %%
churned['streamingmovies'].value_counts()

# %%
churned['contract'].value_counts()

# %%
churned['streamingtv'].value_counts()

# %%
churn_df.groupby('streamingtv')['churn'].value_counts()

# %%
nointer=churn_df[churn_df['streamingtv']=='No internet service']
nointer

# %%
churned=nointer[nointer['churn']=='Yes']

# %%
churned

# %%
churned['phoneservice'].value_counts()

# %%
churn_df['phoneservice'].value_counts()

# %%
# we found that in streaming tv who has no internetservice but having phone service are churning for sure and 100 percent churn rate

# %%
X=churn_df[['tenure','phoneservice','monthlycharges','internetservice','contract','streamingtv','totalcharges'
            ,'techsupport'
           ,'onlinesecurity']]
y=churn_df['churn']
le=LabelEncoder()
oe=OrdinalEncoder()
X=oe.fit_transform(X)
y=le.fit_transform(y)
X=oe.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=14)

# %%
model=BaggingClassifier(LogisticRegression(max_iter=3000),n_estimators=14,max_samples=5200,bootstrap=True)

# %%
model.fit(X_train,y_train)

# %%
model.score(X_test,y_test)

# %%
ConfusionMatrixDisplay.from_estimator(model,X_test,y_test)
plt.show()

# %%
y_predict=model.predict(X_test)
recall_score(y_test,y_predict,pos_label=1)

# %%
precision_score(y_test,y_predict,pos_label=1)

# %%
f1_score(y_test,y_predict,pos_label=1)

# %%
balanced_accuracy_score(y_test,y_predict)

# %%
y_probs=model.predict_proba

# %%
fpr,tpr,threshold=roc_curve(y_test,probs[:,1],pos_label=1)
fpr,tpr,threshold

# %%
auc(fpr,tpr)

# %%
plt.plot(fpr,tpr,marker='o')
plt.show()

# %%



