import os

MODEL_NAME = "gpt-5.4-mini"

try:
    key = os.getenv("OPENAI_API_KEY")
except Exception as e:
    raise RuntimeError("couldn't find openai key")

