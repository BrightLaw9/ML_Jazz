from playsound import playsound

class MusicPlayer: 

    def __init__(self): 
        pass

    def play_music_by_emotion(self, emotion_label): 
        if (emotion_label == 'POSITIVE'): 
            music = '01_Fireworks_Harry_potter.wav'

        playsound(music)

