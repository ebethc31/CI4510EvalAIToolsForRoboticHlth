import whisper
import sounddevice as sd
import numpy as np
import pyttsx3
import time
import threading
from pynput.keyboard import Listener, Key

space_event = threading.Event()

print(sd.query_devices())

sd.default.device = 1

model = whisper.load_model("small.en")

engine = pyttsx3.init()
voices = engine.getProperty('voices')

engine.setProperty('voice', voices[1].id)
engine.setProperty('volume', 1.0)
engine.setProperty('rate', 170)

SAMPLE_RATE = 16000
DURATION = 2
WAKE_WORD = 'eve'

command_library = ["grab marker", "grab pen", "grab pencil", "grab eraser"]

def listen():
    print("Listening...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def transcribe(audio):
    audio = audio.astype(np.float32)
    result = model.transcribe(audio, fp16=False)
    return result["text"].lower().strip()


def speak(text):
    print(f"Robot: {text}")
    engine.say(text)
    engine.runAndWait()

def on_press(key):
    if key == Key.space:
        space_event.set()

def main():

    print("Input detection active. Say 'Trick or Treat' to get candy.")

    listener = Listener(on_press=on_press)
    listener.start()

    while True:
        if space_event.is_set():
            speak("Space bar detected.")
            space_event.clear()
            time.sleep(0.3)
            continue

        audio = listen()
        text = transcribe(audio)
        print(f"Heard: {text}")

        for cmd in command_library:
            if cmd in text:
                speak("Okie dokie artichokie! Here you go!")
                print("grabbin")
                speak("Okay, going back to standby")
                break

            time.sleep(0.1)

            space_event.clear()

            #Placeholder candy grabbing function

            #audio = listen()
            #command = transcribe(audio)
            #print(f"Command: {command}")
            #speak(f"You said: {command}")


        #elif key == Key.space: #FIXME ADJUST SO THAT WHEN SPACE BAR IS PRESSED ONCE THIS BREAKS LOOP
           # speak("Hello! Happy Halloween!")
            #time.sleep(1)

            # Placeholder candy grabbing function

            # audio = listen()
            # command = transcribe(audio)
            # print(f"Command: {command}")
            # speak(f"You said: {command}")

           # speak("Okay, going back to standby.")
           # time.sleep(2)


if __name__ == "__main__":
    main()





#model = whisper.load_model("small")
#result = model.transcribe("WarrenP.mp3")
#print(result["text"])

#comparison = ""