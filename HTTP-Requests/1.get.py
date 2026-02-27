import requests
test = requests.get('https://jsonplaceholder.typicode.com/posts/1')
print(test.json())