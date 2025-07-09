import cv2
import os
import speech_recognition as sr
from googletrans import Translator

DATASET_PATH = "dataset"

LANGUAGES = {
    "english": "en",  
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "hindi": "hi",
    "kannada": "kn"
}

def play_video(video_name):
    video_path = os.path.join(DATASET_PATH, video_name + ".mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Sign Language Video", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f" Warning: No video found for '{video_name}'.")

def recognize_and_translate(language_code):
    recognizer = sr.Recognizer()
    translator = Translator()

    with sr.Microphone() as source:
        print(f" Speak now in {language_code}...")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5)
            spoken_text = recognizer.recognize_google(audio, language=language_code)  
            print(f" Recognized: {spoken_text}")

            if language_code == "en":
                translated_text = spoken_text.lower()  
            else:
                translated_text = translator.translate(spoken_text, src=language_code, dest="en").text.lower()
            
            print(f" Translated to English: {translated_text}")
            return translated_text

        except sr.UnknownValueError:
            print(" Sorry, couldn't recognize speech.")
            return None
        
        except sr.RequestError:
            print(" Speech recognition service error.")
            return None

def main():
    mode = input("\nChoose input mode:\n1️ Text (Type manually in English)\n2️ Speech (Speak in your language)\nEnter 'text' or 'speech': ").strip().lower()

    if mode == "text":
        user_input = input("\n Enter a word in English: ").strip().lower()

    elif mode == "speech":
        print("\n Choose a language:")
        for lang in LANGUAGES:
            print(f"🔹 {lang.capitalize()}")

        chosen_lang = input("\nEnter your language: ").strip().lower()

        if chosen_lang in LANGUAGES:
            user_input = recognize_and_translate(LANGUAGES[chosen_lang])
            if not user_input:
                return
        else:
            print("Invalid language choice!")
            return

    else:
        print("Invalid option! Please enter 'text' or 'speech'.")
        return

    if os.path.exists(os.path.join(DATASET_PATH, user_input + ".mp4")):
        play_video(user_input)
    else:
        print(f"'{user_input}' not found. Playing individual letter videos.")
        for char in user_input.replace(" ", ""): 
            play_video(char)

if __name__ == "__main__":
    main()
