from transformers import pipeline

class AudioClassifier: 

    def __init__(self): 
        self.audio_classifier = pipeline(
            task="audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )

    def classify_emotion(self, audio_file): 
        predictions = self.audio_classifier(audio_file)
        predictions = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in predictions]

        return predictions


    
