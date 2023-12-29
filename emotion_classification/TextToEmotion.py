from transformers import pipeline

class TextClassifier: 

    def __init__(self): 
        self.text_classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
    
    def generate_prediction(self, text_input): 
        print("Predicting...")
        return self.text_classifier(text_input)