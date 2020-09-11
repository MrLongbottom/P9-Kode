import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

from article_extraction import create_article

startdir = '/home/simba/Desktop/Finshed Data 2017'


def convert_process(path):
    article = create_article(path)
    with open("test.json", "a") as file:
        file.write(json.dumps(article, ensure_ascii=False) + '\n')


class Traverse:
    def __init__(self, startpath, n_workers):
        self.startpath = startpath
        self.process_workers = n_workers

    def traverse(self, curpath):
        for dir in os.listdir(curpath):
            if re.fullmatch("[0-9]{4}-[0-9]{2}-[0-9]{2}", dir):
                self.traverse(curpath + "/" + dir)
            elif dir == 'TabletXML':
                self.process_starter(curpath + "/" + dir)
            else:
                print(f"Dir not recognized. {dir}")
                continue

    def process_starter(self, curpath):
        with ProcessPoolExecutor(max_workers=self.process_workers) as executer:
            for dir in os.listdir(curpath):
                print(dir)
                executer.submit(convert_process, curpath + "/" + dir)


if __name__ == '__main__':
    with open("test.json", "w") as file:
        pass
    t = Traverse(startdir, 8)
    t.traverse(startdir)
