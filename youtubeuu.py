from flask import Flask, request, jsonify
import re
import os
import string
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)

# Constants
DATASET_FOLDER = "dataset"
BASE_URL = "http://192.168.3.6:5000/static/videos/"
ALLOWED_EXTENSIONS = {'.mp4'}

# Number word mapping
num_map = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

def extract_video_id(url):
    url = url.split('?')[0]
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_transcript(video_url, lang="en"):
    video_id = extract_video_id(video_url)
    if not video_id:
        return None, "Invalid YouTube URL"
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
    except:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except Exception as e:
            return None, str(e)
    return " ".join([entry['text'] for entry in transcript]), None

def load_dataset():
    return {
        file.split(".mp4")[0].lower()
        for file in os.listdir(DATASET_FOLDER)
        if file.endswith(".mp4")
    }

@app.route("/process", methods=["POST"])
def process_transcript():
    try:
        data = request.json
        url = data.get("url")
        lang = data.get("lang", "en")
        transcript, error = get_transcript(url, lang)

        if not transcript:
            return jsonify({"error": error}), 400

        dataset_words = load_dataset()
        words = transcript.split()
        video_sequence = []
        segments = []

        for word in words:
            clean_word = word.strip(string.punctuation).lower()
            clean_word = num_map.get(clean_word, clean_word)

            if clean_word in dataset_words:
                video_file = f"{clean_word}.mp4"
                video_sequence.append(BASE_URL + video_file)
                segments.append({"word": clean_word, "text": word})
        
        return jsonify({
            "transcript": transcript,
            "segments": segments,
            "video_sequence": video_sequence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)