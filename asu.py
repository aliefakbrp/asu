# def kampret(asu):
#       rmse = [5,6,4,3,43,52]
#       return rmse
# asu=kampret('asu')
# print(asu)
import streamlit as st
# asu.text('asu')
# a = asu.text_input('asu')
# asu.write(a)

# dropdown
option = st.selectbox(
     'How would you like to be contacted?',
     ('Email', 'Home phone', 'Mobile phone'))
st.write('You selected:', option)
