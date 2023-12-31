import pickle 
import streamlit as st

st.set_page_config(
    page_title="Klasifikasi Obesitas"
)
model = pickle.load(open('model.sav', 'rb'))

st.title("Klasifikasi Obesitas")

col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Umur", 11,112,25)
with col2: 
    Gender = st.selectbox("Jenis kelamin", ["Laki-laki", "Perempuan"])

with col1:
    Height = st.number_input("Tinggi badan (Cm)", 120,210,175)
with col2: 
    Weight = st.number_input("Berat badan (Kg)", 10,120,80)

# rumus bmi, digunakan sebagai default value
bmi_result = round((Weight / ((Height / 100)**2)),2)
# Menghindari error min max value pada inputan
if(bmi_result <= 3.9) : bmi_result = 3.9
if(bmi_result >= 37.2)  : bmi_result = 37.2

BMI = st.number_input("Body mass index (berat badan / (tinggi badan)²))", 3.9,37.2,bmi_result)

if(Gender == "Laki-laki") :
    Gender = 1
else : 
    Gender = 2

if st.button("Klasifikasi obesitas"):
    prediction = model.predict([[Age,Gender,Height,Weight,BMI]])
    if (prediction[0] == 1):
        st.info('Kekurangan Berat Badan')
    elif (prediction[0] == 2):
        st.success('Berat Badan Normal')
    elif (prediction[0] == 3):
        st.warning('Kelebihan Berat Badan')
    else:
        st.error('Obesitas')
