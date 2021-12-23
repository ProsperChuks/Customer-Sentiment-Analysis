import requests

url = 'https://localhost:8000/results'
r = requests.post(url, json={'text':'', 'aspect':''})

print(r.json())