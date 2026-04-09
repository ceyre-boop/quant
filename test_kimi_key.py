"""
Simple test of Kimi API key
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("KIMI_API_KEY")
print("=" * 60)
print("KIMI API KEY TEST")
print("=" * 60)
print(f"Key loaded: {'YES' if key else 'NO'}")
print(f"Key length: {len(key) if key else 0}")
print(f"Key prefix: {key[:15]}..." if key else "None")
print(f"Key suffix: ...{key[-10:]}" if key and len(key) > 10 else "None")
print()

# Check format
if key:
    if key.startswith("sk-"):
        print("[OK] Key starts with 'sk-' (correct format)")
    else:
        print("[BAD] Key doesn't start with 'sk-' (wrong format?)")

    if len(key) >= 50:
        print(f"[OK] Key length {len(key)} looks reasonable")
    else:
        print(f"[BAD] Key length {len(key)} seems short")

print()
print("Testing API call...")
print("-" * 60)

# Simple API test
url = "https://api.moonshot.cn/v1/chat/completions"
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
payload = {
    "model": "moonshot-v1-8k",
    "messages": [{"role": "user", "content": "Hello, respond with just: WORKING"}],
    "max_tokens": 20,
    "temperature": 0,
}

try:
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"[SUCCESS] Response: {content}")
    else:
        print(f"[FAILED] Status: {resp.status_code}")
        print(f"Response: {resp.text}")

        # Parse error
        try:
            err = resp.json()
            if "error" in err:
                print(f"Error type: {err['error'].get('type', 'unknown')}")
                print(f"Error message: {err['error'].get('message', 'none')}")
        except:
            pass

except Exception as e:
    print(f"Exception: {e}")

print("-" * 60)
