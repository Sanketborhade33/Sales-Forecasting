import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the XGBoost model
with open('your_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Convert 'Item_Fat_Content' to numerical
    data['Item_Fat_Content'] = data['Item_Fat_Content'].map({'Low Fat': 0, 'Regular': 1})
    # Replace missing values
    data.fillna(data.mean(), inplace=True)  # Replace missing values with mean
    return data

# Function to predict using the model
def predict(data):
    # Preprocess input data
    processed_data = preprocess_input(data)
    # Randomly select a value for 'Outlet_Type'
    processed_data['Outlet_Type'] = np.random.choice([0, 1, 2, 3])
    # Predict
    prediction = model.predict(processed_data)
    return prediction

# Streamlit UI
st.title('Sales Forecasting')

st.write("""
**Item Type:**  
1 = Dairy    \n2 = Soft Drinks    \n3 = Meat    \n4 = Fruits and Vegetables    \n5 = Household    
6 = Baking Goods    \n7 = Snack Foods    \n8 = Frozen Foods    \n9 = BreakFast    \n10 = Health and Hygiene    
11 = Hard Drinks    \n12 = Canned    \n13 = Breads    \n14 = Starchy Foods    \n15 = Others    \n16 = SeaFood

**Outlet Identifier:**  
0 = OUT010    \n1 = OUT013    \n2 = OUT017    \n3 = OUT018    \n4 = OUT019    
5 = OUT027    \n6 = OUT035    \n7 = OUT045    \n8 = OUT046    \n9 = OUT049

**Outlet Size:**  
0 = Small    \n1 = Medium    \n2 = High

**Outlet Location Type:**  
0 = Tier 1    \n1 = Tier 2    \n2 = Tier 3
""")

# Original encoded values
st.write('Enter data for prediction:')
item_identifier = st.number_input("Item Identifier (maximum 1558)")
item_weight = st.number_input('Item Weight')
item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])
item_visibility = st.number_input('Item Visibility')
item_type = st.selectbox('Item Type', list(range(1, 17)))
item_mrp = st.number_input('Item MRP')
outlet_type = st.selectbox("Outlet Type", list(range(4)))
outlet_identifier = st.selectbox('Outlet Identifier', list(range(10)))
outlet_establishment_year = st.number_input('Outlet Establishment Year')
outlet_size = st.selectbox('Outlet Size', list(range(3)))
outlet_location_type = st.selectbox('Outlet Location Type', list(range(3)))

real_sales = st.number_input("Real Sales Value")

data = pd.DataFrame({
    'Item_Identifier': [item_identifier],
    'Item_Weight': [item_weight],
    'Item_Fat_Content': [item_fat_content],
    'Item_Visibility': [item_visibility],
    'Item_Type': [item_type],
    'Item_MRP': [item_mrp],
    'Outlet_Identifier': [outlet_identifier],
    'Outlet_Establishment_Year': [outlet_establishment_year],
    'Outlet_Size': [outlet_size],
    'Outlet_Location_Type': [outlet_location_type]
})

if st.button('Predict'):
    prediction = predict(data)
    st.write('Predicted value:', prediction[0])
    st.write('Real Sales Value:', real_sales)

    # Comparison graph
    comparison_data = pd.DataFrame({
        'Sales': [real_sales, prediction[0]]
    }, index=['Real Sales', 'Predicted Sales'])
    
    st.subheader('Comparison between Real and Predicted Sales')
    fig, ax = plt.subplots()
    comparison_data.plot(kind='bar', ax=ax, color=['skyblue', 'lightgreen'], width=0.5, edgecolor='black')
    ax.set_ylabel('Sales Value')
    ax.set_xlabel('')
    ax.legend(loc='upper right')
    ax.grid(True)
    st.pyplot(fig)






