# text_classification

Exploration of text classification from public datasets, especially emotion labels. See [PDF](./Emotion_Classification.pdf)

# Code

[Load and save data](./emotion_detection_setup.py): load the datasets in use, preprocess them and save split/tokenized versions as .pkl files. Edit main() function to change output

[Train model](./train_emotion_models.py): train several models, need to edit the script to change parameter space and select model

[API](./EmotionAPI/)

[Streamlit App](./EmotionAPI/emotionApp.py)

<!-- # Theory

- There's text classification
- then there is topic modeling (unsupervised)
- But there's also
  - Text-generation | Question-answering
  - Syntax identification | POS tagging
  - [EDA](https://github.com/JasonKessler/scattertext)
- Challenges
  - How to remove bias from the dataset?
    - [Embedding](https://developers.googleblog.com/2018/04/text-embedding-models-contain-bias.html) models contain bias. To overcome this we have to document exactly which terms to target and either:
      - identify the sources of bias in the data and leave them out (this would be apply to the case where we use TF-IDF)
      - reduce bias by [projecting](https://arxiv.org/abs/1607.06520) the embeddings to neturalize it
  - Challenges with NLP on non-English data? How to mitigate?
    - Locate a giant corpus of non-English text and train new word-embeddings
    - Use open-sourced pre-trained [embeddings](https://github.com/Babylonpartners/fastText_multilingual) -->
