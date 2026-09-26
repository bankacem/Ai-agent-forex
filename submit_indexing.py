import json
import urllib.request
import urllib.parse
import urllib.error
import sys
import os

urls = [
    # Article 1: Gemini in Chrome
    "https://extensionto.com/blog/gemini-in-chrome-2026-complete-guide",
    "https://extensionto.com/fr/blog/gemini-in-chrome-2026-complete-guide",
    "https://extensionto.com/es/blog/gemini-in-chrome-2026-complete-guide",
    "https://extensionto.com/pt/blog/gemini-in-chrome-2026-complete-guide",
    "https://extensionto.com/ar/blog/gemini-in-chrome-2026-complete-guide",
    # Article 2: Best AI Extensions
    "https://extensionto.com/blog/best-ai-chrome-extensions-2026",
    "https://extensionto.com/fr/blog/best-ai-chrome-extensions-2026",
    "https://extensionto.com/es/blog/best-ai-chrome-extensions-2026",
    "https://extensionto.com/pt/blog/best-ai-chrome-extensions-2026",
    "https://extensionto.com/ar/blog/best-ai-chrome-extensions-2026",
]

def check_url(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Search Engine Indexing Checker)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, "OK"
    except urllib.error.HTTPError as e:
        return e.code, f"HTTPError: {e.reason}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_indexnow_payload(api_key=None):
    key = api_key or os.environ.get("INDEXNOW_KEY", "YOUR_INDEXNOW_KEY")
    payload = {
        "host": "extensionto.com",
        "key": key,
        "keyLocation": f"https://extensionto.com/{key}.txt",
        "urlList": urls
    }
    return payload

def submit_to_indexnow(payload, endpoint="https://api.indexnow.org/indexnow"):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "IndexNow-Submitter/1.0"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, "Submitted successfully"
    except urllib.error.HTTPError as e:
        return e.code, f"HTTPError: {e.reason}"
    except Exception as e:
        return None, f"Error: {str(e)}"

if __name__ == "__main__":
    print("--- Checking HTTP Status for 10 URLs ---")
    for url in urls:
        status, msg = check_url(url)
        print(f"[{status}] {url} -> {msg}")

    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    payload = generate_indexnow_payload(api_key)
    payload_json = json.dumps(payload, indent=2)

    with open("indexnow_payload.json", "w") as f:
        f.write(payload_json)
    print("\n--- IndexNow Payload Generated ---")
    print(payload_json)

    if api_key or "INDEXNOW_KEY" in os.environ:
        print("\n--- Submitting to IndexNow Endpoint ---")
        status, msg = submit_to_indexnow(payload)
        print(f"IndexNow submission result: [{status}] {msg}")
    else:
        print("\nNote: Provide your IndexNow API key as argument or via INDEXNOW_KEY env var to transmit live submission.")
