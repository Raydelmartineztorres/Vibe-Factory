import os
import requests

API_KEY = os.getenv("QWEEN3_API_KEY")

def fix_ui(prompt: str) -> str:
    """Send a prompt to Qween‑3 and return the generated CSS/JS.
    The function expects the environment variable QWEEN3_API_KEY to be set.
    """
    if not API_KEY:
        raise RuntimeError("QWEEN3_API_KEY not set in environment")
    response = requests.post(
        "https://api.qween3.com/v1/completions",
        json={"prompt": prompt, "max_tokens": 800},
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    # API returns a list of choices; we take the first one
    return data.get("choices", [{}])[0].get("text", "")

if __name__ == "__main__":
    # Simple CLI for testing
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qween3_wrapper.py '<prompt>'")
        sys.exit(1)
    prompt = sys.argv[1]
    print(fix_ui(prompt))
