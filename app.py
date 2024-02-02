import streamlit as st
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

st.title('Admission Data Science Project')



Gre_Score = st.number_input('Gre Score')
st.write('The current Gre Score is ', Gre_Score)
Tofel_Score = st.number_input('Tofel_Score')
st.write('The current Tofel_Score is ', Tofel_Score)

University_Rating=st.selectbox('University Rating',
    (1, 2, 3,4,5))

SOP=st.number_input('SOP',min_value=0.00,max_value=5.00)
LOR=st.number_input("LOR",min_value=1.00,max_value=5.00)
CGPA=st.number_input("CGPA",min_value=0.00,max_value=10.00)
Research=st.selectbox('Research',
    (1, 0))

user_input=[[Gre_Score,Tofel_Score,University_Rating,SOP,LOR,CGPA,Research]]
sc.fit(x)
scaled_user_input=sc.transform((user_input))

##st.write(user_input)
##st.write(scaled_user_input)
loaded_model=pickle.load(open('lr_for_admission','rb'))
result=loaded_model.predict(scaled_user_input)
st.write(result)

if st.button("Predict"):
    result_percentage= result*100
    st.header("Percentage Of you Getting Admitted In University Is"+str(result_percentage))