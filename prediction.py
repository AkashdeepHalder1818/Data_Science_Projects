import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv('Admission_Prediction.csv')
df['GRE Score'].fillna(df['GRE Score'].mean(),inplace=True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mean(),inplace=True)
df['University Rating'].fillna(df['University Rating'].mean(),inplace=True)
df.drop(columns=['Serial No.'],inplace=True)

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
sc = StandardScaler()
x= sc.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

lr=LinearRegression()
lr.fit(x_train,y_train)

filename='lr_for_admission'
pickle.dump(lr,open(filename,'wb'))
loaded_model=pickle.load(open('lr_for_admission','rb'))
result=loaded_model.score(x_test,y_test)
print(result)