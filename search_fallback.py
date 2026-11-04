# search_fallback.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

def get_google_snippets(query: str, num_results: int = 3) -> str:
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": num_results
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        snippets = []
        for item in data.get("items", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            snippets.append(f"\n **{title}**\n{snippet}\n {link}")

        return "\n\n".join(snippets) if snippets else "No relevant information found."

    except Exception as e:
        return f"[Google Search Error] {e}"

# Support structure for explainability output in the UI:
# Each model should output:
# - Response
# - Helpfulness, Correctness, Coherence, Tone, Bias
# - Reasoning/Explanation why the score was assigned
#
# Radar Chart Inputs Example:
# scores = {
#   'Model': 'GPT-4' (see model_config.MODEL_KEY_*),
#   'Helpfulness': 0.8,
#   'Correctness': 0.75,
#   'Coherence': 0.85,
#   'Tone': 0.7,
# }

# CSV export format should include:
# model, response, helpfulness, correctness, coherence, tone, bias_flag, reasoning, source_info

# Charts and UI logic should be implemented in gradio_full_llm_eval.py using Plotly or Matplotlib
