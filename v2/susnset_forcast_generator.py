import asyncio
import json
from langchain_openai import ChatOpenAI
from prompt import prompt_generator
from config import key, MODEL_NAME
from get_forecast import main
TEL_AVIV_LAT = 32.087543478872
TEL_AVIV_LON = 34.7704678
TEL_AVIV_TIMEZONE = "Asia/Jerusalem"
requests_list = [
    {"lat":TEL_AVIV_LAT, "lon":TEL_AVIV_LON, "tz":TEL_AVIV_TIMEZONE}
]

llm = ChatOpenAI(
    api_key=key,
    model=MODEL_NAME,
    temperature=0.4,
    max_tokens= 400,
)


def generate_forecast_text() -> str:
    forecast = asyncio.run(main(requests_list))

    prompt = prompt_generator(forecast)

    response = llm.invoke(prompt)

    raw_content = response.content.strip()

    try:
        parsed = json.loads(raw_content)
        return parsed["body"]
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON returned by model:\n{raw_content}")


if __name__ == "__main__":
    print(generate_forecast_text())