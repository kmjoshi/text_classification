# emotion-classification RESTful API and webapp

- how to run: 
    - install python>=3.5
    - pip install -r requirements.txt
    - python insert_file_name.py

- [Flask command-line API](emotionAPI.py): 
    - use command: curl -X GET http://127.0.0.1:5000/ -d query='I love learning!!'
    - [resource](https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166)

- [Flask web-app API](emotion_detection_API.py):
    - enter text into field box and press submit to view JSON of emotion values. Navigate with back button to test more entries
    - [resource](https://www.datacamp.com/community/tutorials/machine-learning-models-api-python)
