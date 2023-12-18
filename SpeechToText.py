import speech_recognition as sr


class MicSpeechToText: 
    
    def __init__(self): 
        self.mic = sr.Microphone()
        self.recognizer = sr.Recognizer()

    def get_audio(self): 
        return self.recognizer.listen(self.mic)

    def get_text_from_speech(self): 
        # Uses the Google Speech Recognition model 
        
        with self.mic: 
            #Get audio
            print("Say something!")
            audio = self.recognizer.listen(self.mic)

            #Interpret
            return self.recognizer.recognize_google(audio)


