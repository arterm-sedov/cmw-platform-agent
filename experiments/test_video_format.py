"""
Test video_url format with OpenRouter API
"""
import os
import base64
import requests

video_path = r'E:\Downloads\video_2026-02-10_16-56-58.mp4'
api_key = 'XXX'

print('Reading video...')
with open(video_path, 'rb') as f:
    video_data = base64.b64encode(f.read()).decode('utf-8')

data_url = f'data:video/mp4;base64,{video_data}'
print(f'Video encoded: {len(video_data)} chars, {len(video_data)*3/4/1024/1024:.1f}MB')

payload = {
    'model': 'qwen/qwen3.6-plus',
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Describe what is happening in this video.'},
            {'type': 'video_url', 'video_url': {'url': data_url}}
        ]
    }]
}

print('Sending to OpenRouter with video_url format...')
response = requests.post(
    'https://openrouter.ai/api/v1/chat/completions',
    headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
    json=payload, timeout=120
)

print(f'Status: {response.status_code}')
data = response.json()

if response.ok:
    print('\n✅ SUCCESS!')
    print(data['choices'][0]['message']['content'][:500])
else:
    print(f'\n❌ ERROR: {data}')
