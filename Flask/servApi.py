import re
import os
from pathlib import Path

from flask import Flask, request, send_file

app = Flask(__name__)
_path = "/media/theis/Thomas' Expansion Drive/Pruned"


@app.route('/')
def test():
    return "Flask is running"


@app.route('/getimage/<path:path>', methods=['GET'])
def test1(path):
    path = re.sub(".jp2", ".jpg", path)
    fullpath = None
    for dir in os.listdir(_path):
        path1 = _path + "/" + dir
        if os.path.isdir(path1):
            tmp = path1 + "/" + path
            image = Path(tmp)
            if image.exists():
                fullpath = tmp
                break

    print(f"This is the full path {fullpath}")

    return send_file(filename_or_fp=fullpath, as_attachment=True)
