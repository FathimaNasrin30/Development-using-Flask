from flask import Flask, request, render_template
from EmotionDetection import emotion_detector

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detection():
    """
    This function handles user input from the webpage
    and returns formatted emotion analysis.
    """
    text_to_analyze = request.args.get('textToAnalyze')

    # Call emotion detector function
    response = emotion_detector(text_to_analyze)

    # Error handling for empty or invalid input
    if response['dominant_emotion'] is None:
        return "Invalid text! Please try again!"

    # Format output
    result = (
        "For the given statement, the system response is <br>"
        f"anger: {response['anger']} <br>"
        f"disgust: {response['disgust']} <br>"
        f"fear: {response['fear']} <br>"
        f"joy: {response['joy']} <br>"
        f"sadness: {response['sadness']} <br>"
        f"<b>The dominant emotion is {response['dominant_emotion']}</b>"
    )

    return result


if __name__ == "__main__":
    app.run(debug=True)
