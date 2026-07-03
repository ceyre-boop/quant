import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('KIMI_API_KEY')

print("Testing new K2/K2.5 models with new key format...")
print(f"Key: {key[:20]}...")
print()

models = [
    'kimi-k2',
    'kimi-k2.5', 
    'kimi-latest',
    'moonshot-v1-8k',  # Old model
]

headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

for model in models:
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'Say: WORKING'}],
        'max_tokens': 10
    }
    resp = requests.post(
        'https://api.moonshot.cn/v1/chat/completions',
        headers=headers,
        json=payload,
        timeout=15
    )
    status = 'SUCCESS!' if resp.status_code == 200 else resp.status_code
    print(f'{model}: {status}')
    if resp.status_code == 200:
        content = resp.json()['choices'][0]['message']['content']
        print(f'  Response: {content}')
        print()
        print('>>> FOUND WORKING MODEL! <<<')
        break
    print()
