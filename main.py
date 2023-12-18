import SpeechToText
import ClassifyEmotion
import TextToEmotion
import Music

#Sound output imports


def main(): 

    speech_to_text = SpeechToText.MicSpeechToText()

    emotion_classifier = TextToEmotion.TextClassifier()

    music = ''

    music_player = Music.MusicPlayer() 

    while (True):

        text = speech_to_text.get_text_from_speech()

        #audio = speech_to_text.get_audio()

        emotion = emotion_classifier.generate_prediction(text)

        print(f"Text: {text}")
        print(f"Emotion: {emotion}")

        emotion_label = emotion[0].get('label') 

        music_player.play_music_by_emotion(emotion_label)



if __name__ == '__main__': 
    main()