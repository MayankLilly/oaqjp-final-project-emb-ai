import json
import requests


def emotion_detector(text_to_analyze):
    """Analyze emotion text and return formatted emotion scores with dominant emotion."""
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return None

    if response.status_code == 400:
        return None

    formatted_response = json.loads(response.text)

    emotion_data = (
        formatted_response.get("emotionPredictions", [{}])[0]
        .get("emotion", {})
    )

    result = {
        "anger": emotion_data.get("anger", 0.0),
        "disgust": emotion_data.get("disgust", 0.0),
        "fear": emotion_data.get("fear", 0.0),
        "joy": emotion_data.get("joy", 0.0),
        "sadness": emotion_data.get("sadness", 0.0),
    }
    result["dominant_emotion"] = max(result, key=result.get)

    return result
