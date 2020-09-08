import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests

def post_json(path):
    filenames = [path + "/" + file for file in os.listdir(path) if file.endswith(".json")]
    total = len(filenames)
    print(path)
    with ThreadPoolExecutor(max_workers=8) as executer:
        for filename in filenames:
            executer.submit(poster, filename)

def poster(filename):
    with open(filename, 'rb') as f:
        headers = {'Content-type': 'application/json'}
        requests.post('http://localhost:8983/solr/article/update?commit=true', headers=headers, data=f.read())
    print(f'Posted {filename}')
# USE TO EMPTY SOLR DB: curl http://localhost:8983/solr/article/update?commit=true -H "Content-Type: text/xml" --data-binary '<delete><query>*:*</query></delete>'