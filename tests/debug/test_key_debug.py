# WARNING: Debug/developer script only. Do NOT run in CI or production.
# Key diagnostic prints are intentional for manual debugging; never share output.
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("KIMI_API_KEY")

print("Key Diagnostics:")
print(f"  Length: {len(key)}")
print(f"  Starts with 'sk-': {key.startswith('sk-')}")
print(f"  Contains 'kimi': {'kimi' in key}")
print(f"  Has newline: {chr(10) in key}")
print(f"  Has spaces at start: {key[0] == ' ' if key else False}")
print(f"  Has spaces at end: {key[-1] == ' ' if key else False}")
print(f"  Printable: {all(32 <= ord(c) <= 126 for c in key)}")

# Show hex of first/last 10 chars
print(f"  First 10 hex: {' '.join(f'{ord(c):02x}' for c in key[:10])}")
print(f"  Last 10 hex: {' '.join(f'{ord(c):02x}' for c in key[-10:])}")
