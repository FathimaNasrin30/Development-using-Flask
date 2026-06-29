#!/usr/bin/env python3
"""
emotiondetection.py

Text-based emotion detection using IBM Watson Natural Language Understanding.

Usage (CLI):
  - Using environment variables:
      export WATSON_NLU_APIKEY="your-apikey"
      export WATSON_NLU_URL="https://api.us-south.natural-language-understanding.watson.cloud.ibm.com"
      python emotiondetection.py --text "I am so happy today!"

  - Passing API credentials on the command line:
      python emotiondetection.py --text "I'm upset" --apikey YOUR_KEY --url YOUR_URL

  - Analyze a file:
      python emotiondetection.py --file ./samples/utterance.txt

Returns JSON with emotion scores and prints the top emotion.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional

try:
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
except Exception as e:
    raise ImportError(
        "Missing dependency: install with `pip install ibm-watson python-dotenv` "
        "and ensure network access to IBM Cloud." ) from e


DEFAULT_VERSION = "2021-08-01"
ENV_APIKEY = "WATSON_NLU_APIKEY"
ENV_URL = "WATSON_NLU_URL"


def _build_nlu_client(apikey: str, url: str) -> NaturalLanguageUnderstandingV1:
    authenticator = IAMAuthenticator(apikey)
    nlu = NaturalLanguageUnderstandingV1(version=DEFAULT_VERSION, authenticator=authenticator)
    nlu.set_service_url(url)
    return nlu


def analyze_emotion(
    text: str,
    apikey: Optional[str] = None,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze emotions in the provided text using IBM Watson NLU.

    Returns the raw API result and a simplified summary:
      {
        "raw": {...},
        "document_emotion": {"joy": 0.1, "sadness": 0.2, ...},
        "top_emotion": ("sadness", 0.2)
      }

    Credentials:
      - If apikey or url are not provided, the function reads WATSON_NLU_APIKEY and WATSON_NLU_URL from env.
    """
    apikey = apikey or os.getenv(ENV_APIKEY)
    url = url or os.getenv(ENV_URL)

    if not apikey or not url:
        raise ValueError(
            f"API credentials missing. Set env vars {ENV_APIKEY} and {ENV_URL} "
            "or pass --apikey/--url to the script."
        )

    nlu = _build_nlu_client(apikey, url)

    response = nlu.analyze(
        text=text,
        features=Features(emotion=EmotionOptions(document=True))
    ).get_result()

    # Navigate the response safely
    doc_emotion = {}
    try:
        doc_emotion = response.get("emotion", {}).get("document", {}).get("emotion", {}) or {}
    except Exception:
        doc_emotion = {}

    # Determine top emotion
    top_emotion = None
    if doc_emotion:
        top_emotion = max(doc_emotion.items(), key=lambda kv: kv[1])

    return {
        "raw": response,
        "document_emotion": doc_emotion,
        "top_emotion": top_emotion,
    }


def _load_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _pretty_print_result(result: Dict[str, Any], json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    doc_emotion = result.get("document_emotion", {})
    top = result.get("top_emotion")
    print("Emotion scores (document):")
    for k, v in sorted(doc_emotion.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s}: {v:.4f}")
    if top:
        print(f"\nTop emotion: {top[0]} ({top[1]:.4f})")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Emotion detection using IBM Watson NLU")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", help="Text to analyze", type=str)
    group.add_argument("--file", "-f", help="Path to text file to analyze", type=str)

    parser.add_argument("--apikey", help="IBM Watson NLU API key (overrides env var)", type=str)
    parser.add_argument("--url", help="IBM Watson NLU service URL (overrides env var)", type=str)
    parser.add_argument("--json", action="store_true", help="Output full JSON result")
    args = parser.parse_args(argv)

    if args.file:
        text = _load_text_from_file(args.file)
    else:
        text = args.text

    try:
        result = analyze_emotion(text, apikey=args.apikey, url=args.url)
        _pretty_print_result(result, json_output=args.json)
    except Exception as exc:
        print("Error:", str(exc), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
