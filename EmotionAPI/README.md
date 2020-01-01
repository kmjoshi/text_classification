# emotion-classification RESTful API and webapp

- how to run locally: 
    - install python>=3.5
    - pip install -r requirements.txt
    - API: python emotionAPI.py
    - Streamlit app: streamlit run emotionApp.py
        - needs running API

- [Flask command-line API](emotionAPI.py): 
    - use command: curl -X GET http://127.0.0.1:5000/ -d query='I love learning!!'
    - [ref 1](https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166) [ref 2](https://www.datacamp.com/community/tutorials/machine-learning-models-api-python)

- [Streamlit web-app](emotionApp.py):
    - enter text and view the predicted emotion values
    - [ref](https://github.com/kmjoshi/intro_to_streamlit)
