import streamlit as st
import requests

st.title('Big-5 Emotion Classification')
st.write('This widget queries the API and returns response below. Each emotion is rated on a scale of 0-100.')

query = st.text_input('Enter any emotion-filled text here!', 'There are absolutely no emotions in this text whatsoever!')
response = requests.get('http://localhost:5000', params={'query': query})

st.json(response.json())

st.markdown('Find slides of how the model was designed [here](https://github.com/kmjoshi/text_classification/blob/master/Emotion_Classification.pdf)')