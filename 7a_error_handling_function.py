import requests

def emotion_detector(text):
    """
    This function sends text to Watson NLP API and returns
    emotion analysis with proper error handling.
    """

    # Handle empty input
    if text is None or text.strip() == "":
        return {
            "status_code": 400,
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {
        "raw_document": {
            "text": text
        }
    }

    response = requests.post(url, json=payload, headers=headers)

    # Handle API failure or bad response
    if response.status_code != 200:
        return {
            "status_code": 400,
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    response_json = response.json()
    emotions = response_json['emotionPredictions'][0]['emotion']

    # Extract emotions
    anger = emotions['anger']
    disgust = emotions['disgust']
    fear = emotions['fear']
    joy = emotions['joy']
    sadness = emotions['sadness']

    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "status_code": 200,
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion
    }
