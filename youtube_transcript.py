import os
import re
import string
from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi

num_map = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

app = Flask(__name__, static_folder="dataset", static_url_path="/sign_videos")
CORS(app)

def extract_video_id(url):
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})", url)
    return m.group(1) if m else None

def get_transcript(url, lang="en"):
    vid = extract_video_id(url)
    if not vid: return None
    try:
        data = YouTubeTranscriptApi.get_transcript(vid, languages=[lang])
        return " ".join(e["text"] for e in data)
    except:
        return None

def load_dataset():
    video_folder = app.static_folder
    available = set()
    for f in os.listdir(video_folder):
        if f.lower().endswith(".mp4"):
            file_path = os.path.join(video_folder, f)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:  # Check if file exists and is >1KB
                available.add(f[:-4])  # Add filename without .mp4 extension
    return available

@app.route("/process", methods=["POST"])
def process():
    js = request.get_json()
    text = get_transcript(js.get("url", ""), js.get("lang", "en"))
    if not text:
        return jsonify(error="no transcript"), 400

    words = text.split()
    # break into 5-word subtitles
    subs = [" ".join(words[i:i+5]) for i in range(0, len(words), 5)]
    available = load_dataset()

    base = request.host_url.rstrip("/")
    video_folder = app.static_folder
    urls = []
    for w in words:
        c = w.strip(string.punctuation).lower()
        c = num_map.get(c, c)
        video_path = f"{video_folder}/{c}.mp4"
        if c in available and os.path.exists(video_path) and os.path.getsize(video_path) > 1024:
            urls.append(f"{base}/sign_videos/{c}.mp4")

    return jsonify(sentences=subs, video_urls=urls)

if __name__ == "__main__":
    app.run(debug=True)