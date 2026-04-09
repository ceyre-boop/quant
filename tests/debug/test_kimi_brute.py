"""
Brute force test all Kimi API configurations
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("KIMI_API_KEY")

print("=" * 70)
print("KIMI API BRUTE FORCE TEST")
print("=" * 70)
print(f"Key: {key[:20]}...")
print()

# Test every possible configuration
configs = [
    # Standard
    {
        "name": "Standard Bearer",
        "url": "https://api.moonshot.cn/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        "payload": {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
    },
    # No Bearer prefix
    {
        "name": "No Bearer prefix",
        "url": "https://api.moonshot.cn/v1/chat/completions",
        "headers": {"Authorization": key, "Content-Type": "application/json"},
        "payload": {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
    },
    # API-Key header
    {
        "name": "X-API-Key header",
        "url": "https://api.moonshot.cn/v1/chat/completions",
        "headers": {"X-API-Key": key, "Content-Type": "application/json"},
        "payload": {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
    },
    # api-key header (lowercase)
    {
        "name": "api-key header",
        "url": "https://api.moonshot.cn/v1/chat/completions",
        "headers": {"api-key": key, "Content-Type": "application/json"},
        "payload": {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
    },
    # Different model
    {
        "name": "moonshot-v1-32k model",
        "url": "https://api.moonshot.cn/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        "payload": {
            "model": "moonshot-v1-32k",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
    },
    # kimi-latest model
    {
        "name": "kimi-latest model",
        "url": "https://api.moonshot.cn/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        "payload": {
            "model": "kimi-latest",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
    },
    # HTTP instead of HTTPS
    {
        "name": "HTTP instead of HTTPS",
        "url": "http://api.moonshot.cn/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        "payload": {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
    },
    # Without /v1
    {
        "name": "Without /v1 prefix",
        "url": "https://api.moonshot.cn/chat/completions",
        "headers": {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        "payload": {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
    },
]

for i, config in enumerate(configs, 1):
    print(f"\nTest {i}: {config['name']}")
    print(f"  URL: {config['url']}")
    try:
        resp = requests.post(
            config["url"],
            headers=config["headers"],
            json=config["payload"],
            timeout=15,
            allow_redirects=True,
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"  >>> SUCCESS! <<<")
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            print(f"  Response: {content}")
            break
        else:
            err = resp.json().get("error", {})
            print(f"  Error: {err.get('message', 'Unknown')}")
    except Exception as e:
        print(f"  Exception: {str(e)[:60]}")

print("\n" + "=" * 70)
print("Tests complete")
print("=" * 70)
