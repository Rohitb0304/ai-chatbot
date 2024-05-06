from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import google.generativeai as genai

app = Flask(__name__)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json.get("user_input")

    # Perform speech recognition if the user input starts with "data:audio"
    if user_input.startswith("data:audio"):
        recognized_text = recognize_speech(user_input)
        response = generate_response(recognized_text)
    else:
        response = generate_response(user_input)

    return jsonify({"response": response})

def recognize_speech(audio_data):
    r = sr.Recognizer()
    audio_bytes = audio_data.split(",")[1].encode()  # Extract audio data from base64 string
    with sr.AudioData(audio_bytes) as source:
        audio = r.record(source)
    try:
        recognized_text = r.recognize_google(audio)
        return recognized_text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        return "Speech recognition service unavailable."

def generate_response(input_text):
    genai.configure(api_key="Enter API KEY Here")  

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest", generation_config=generation_config, safety_settings=safety_settings
    )

    response = model.generate_content([input_text + "\n\n"])
    return response.text.strip()

if __name__ == "__main__":
    app.run(debug=True)
