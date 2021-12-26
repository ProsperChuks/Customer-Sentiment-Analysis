import requests

url = 'http://absa.eu-west-2.elasticbeanstalk.com/results'
r = requests.post(url, json={'text':'the app is bad', 'aspect':'app'})

print(r.json())