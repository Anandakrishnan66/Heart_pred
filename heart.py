import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sn


disease_df=pd.read_csv("./framingham.csv")
disease_df.drop(['education'],inplace=True,axis=1)
disease_df.rename(columns={'male':'Sex_male'},inplace=True)

disease_df.dropna(axis=0,inplace=True)
# print(disease_df.head(),disease_df.shape)
# print(disease_df.TenYearCHD.value_counts())


x=np.asarray(disease_df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose']])
y=np.asarray(disease_df['TenYearCHD'])

x=preprocessing.StandardScaler().fit(x).transform(x)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=4)

# print(X_train.shape)

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train,Y_train)
y_pred=logreg.predict(X_test)

from sklearn.metrics import jaccard_score
print(" ")
print("Accurace of model in jaccard similarity score is = ", jaccard_score(Y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(X_train,Y_train)
y_pred= rf.predict(X_test)

score=rf.score(X_test,Y_test)*100
print('Accuracy  rf model i s', score)


from sklearn.metrics import confusion_matrix ,classification_report

cm=confusion_matrix(Y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))

sn.heatmap(conf_matrix,annot=True,fmt='d',cmap="Greens")


plt.show()


print('The details for confusion matrix is =')
print (classification_report(Y_test, y_pred))
