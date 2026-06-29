from flask import Flask, request, render_template
from EmotionDetection import emotion_detector

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detection():
    """
    Handles user input and detects emotion with error handling
    for blank input.
    """

    text_to_analyze = request.args.get('textToAnalyze')

    # Handle blank input directly
    if text_to_analyze is None or text_to_analyze.strip() == "":
        return "Invalid text! Please try again!", 400

    # Call emotion detector
    result = emotion_detector(text_to_analyze)

    # Handle case where API fails or returns invalid response
    if result["dominant_emotion"] is None:
        return "Invalid text! Please try again!", 400

    # Format output
    response = (
        "For the given statement, the system response is <br>"
        f"anger: {result['anger']} <br>"
        f"disgust: {result['disgust']} <br>"
        f"fear: {result['fear']} <br>"
        f"joy: {result['joy']} <br>"
        f"sadness: {result['sadness']} <br>"
        f"<b>The dominant emotion is {result['dominant_emotion']}</b>"
    )

    return response, 200


if __name__ == "__main__":
    app.run(debug=True)
