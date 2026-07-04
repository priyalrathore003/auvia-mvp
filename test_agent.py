import requests
import base64
import json

# Generate a small mock base64 string representing dummy audio data
dummy_audio_bytes = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09" * 100
mock_audio_base64 = base64.b64encode(dummy_audio_bytes).decode("utf-8")

payload = {
    "query": "First, query the audio transcripts for 'File_A' to identify its specific audio issue. Then, based on the transcript's description of the problem, run the noise reduction filter using the provided base64 audio string to fix it.",
    "audio_b64": mock_audio_base64,
    "context": {
        "audio_b64": mock_audio_base64,
        "format": "wav",
        "description": "Mock dummy audio data for testing noise reduction tool"
    }
}

print("Sending POST request to http://localhost:8000/process-audio ...")
print(f"Payload query: {payload['query']}")
print(f"audio_b64 key present: {'audio_b64' in payload}")
print(f"Audio base64 (first 60 chars): {mock_audio_base64[:60]}...")

try:
    response = requests.post(
        "http://localhost:8000/process-audio",
        json=payload,
        timeout=30
    )
    print(f"\nStatus Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.ConnectionError as e:
    print(f"\nConnection Error: {e}")
    print("Make sure the FastAPI server is running on http://localhost:8000")
except requests.exceptions.Timeout:
    print("\nRequest timed out after 30 seconds.")
except Exception as e:
    print(f"\nUnexpected error: {e}")