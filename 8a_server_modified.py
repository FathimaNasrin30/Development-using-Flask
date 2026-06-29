"""
Flask server for Emotion Detection application.
Performs emotion analysis and returns formatted results.
"""

from flask import Flask, request, render_template
from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/")
def home():
    """
    Render the home page.
    """
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detection():
    """
    Handle emotion detection requests from the user.
    Validates input and returns formatted output.
    """
    text_to_analyze = request.args.get('textToAnalyze')

    # Handle blank input
    if text_to_analyze is None or text_to_analyze.strip() == "":
        return "Invalid text! Please try again!", 400

    # Call emotion detection function
    result = emotion_detector(text_to_analyze)

    # Handle API or processing errors
    if result["dominant_emotion"] is None:
        return "Invalid text! Please try again!", 400

    # Format response for display
    formatted_response = (
        "For the given statement, the system response is <br>"
        f"anger: {result['anger']} <br>"
        f"disgust: {result['disgust']} <br>"
        f"fear: {result['fear']} <br>"
        f"joy: {result['joy']} <br>"
        f"sadness: {result['sadness']} <br>"
        f"<b>The dominant emotion is {result['dominant_emotion']}</b>"
    )

    return formatted_response, 200


if __name__ == "__main__":
    app.run(debug=True)
``
