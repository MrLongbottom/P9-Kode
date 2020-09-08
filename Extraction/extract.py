import glob, os
import xml.etree.ElementTree as ET
import re
from math import floor
from Extraction.text import FileStruct, TextBlock, TextLine
 
class Extract:
    def __init__(self, dir_):
        self._dir_ = dir_

    def run(self):
        files = []
        filenames = [self._dir_ + "/" + file for file in os.listdir(self._dir_) if file.endswith(".alto.xml")]
        for filename in filenames:
            files.append(self._parse_file(filename))

        return files

    def _parse_file(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        prefix = re.sub('}alto', '}', root.tag)

        file = FileStruct(os.path.abspath(filename), list(root.iter(prefix + "fileName"))[0].text)
    
        for block in root.iter(prefix + "TextBlock"):
            
            file.add_txtblock(self._parse_block(block, prefix))

        return file
                
    def _parse_block(self, block, prefix):
        attrib_ = block.attrib
        
        txtblock = TextBlock(attrib_["ID"], int(attrib_["HEIGHT"]), 
                            int(attrib_["WIDTH"]), 
                            int(attrib_["HPOS"]), 
                            int(attrib_["VPOS"]))

        for line in block.findall(prefix + "TextLine"):
            txtblock.add_line(self._parse_line(line, prefix))
            
        return txtblock

    def _parse_line(self, line, prefix):
        attrib_ = line.attrib
        txtline = TextLine(attrib_["ID"], self._get_style_val(attrib_["STYLEREFS"]), attrib_["HEIGHT"], attrib_["WIDTH"], attrib_["HPOS"], attrib_["VPOS"])

        for child in line.findall(prefix + "String"):
            txtline.add_word(child.attrib["CONTENT"])

        return txtline

    def _get_style_val(self, style):
        return float(re.sub("TS", "", style))
