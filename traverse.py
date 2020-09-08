import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import Misc.converter as converter
import Misc.post_json as poster
import Extraction.construct as constructer

# Add Arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", help="path to directory to start traversal.")
ap.add_argument("-s", "--skip", help="processses to skip in format [chars]. 'p' skip post, 'e' skip extraction. 'c' skip image convertion (also pruning).")

#  Handle arguments.
args = vars(ap.parse_args())

if args["directory"]:
    startdir = args["directory"]
else:
    exit("Please enter starting directory.")

if args["skip"]:
    toskip = args["skip"]

class Traverse:
    def __init__(self, startpath, toskip):
        self.startpath = startpath
        if 'p' in toskip:
            self.process_workers = 8
            self.skip_posting = True
        else:
            self.skip_posting = False
            self.process_workers = 4

        if 'e' in toskip:
            self.skip_extraction = True
        else:
            self.skip_extraction = False

        if 'c' in toskip:
            self.skip_conversion = True
        else:
            self.skip_conversion = False

    def traverse(self, curpath):
        for dir in os.listdir(curpath):
            if re.fullmatch("B[0-9]{12}-RT[0-9]", dir):
                print(dir)
                self.traverse(curpath + "/" + dir)
            elif re.fullmatch("[0-9]{12}-[0-9]{2}", dir):
                print(dir)
                self.process_starter(curpath + "/" + dir)
            else:
                print(f"Dir not recognized. {dir}")
                continue

    def process_starter(self, curpath):
        with ProcessPoolExecutor(max_workers=self.process_workers) as executer:
            for dir in os.listdir(curpath):
                if re.fullmatch("[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}", dir):
                    executer.submit(self.process, curpath + "/" + dir)
                else:
                    continue

    def process(self, path):
        with ProcessPoolExecutor(max_workers=2) as executer:
            if not self.skip_conversion:
                executer.submit(self.convert_process, path)

            executer.submit(self.data_process, path)

    def data_process(self, path):
        if not self.skip_extraction:
            c = constructer.Construct()
            c.construct(path)

        if not self.skip_posting:
            poster.post_json(path)

    def convert_process(self, path):
        converter.convertfiles(path)
        self.remove_files(path)

    def remove_files(self, path):
        for file in [path + "/" + file for file in os.listdir(path) if self.to_remove(file)]:
            os.remove(file)


    def to_remove(self, file):
        return (file.endswith(".xml") and not file.endswith(".alto.xml")) or file.endswith(".md5") or file.endswith(".jp2")



t = Traverse(startdir, toskip)
t.traverse(startdir)