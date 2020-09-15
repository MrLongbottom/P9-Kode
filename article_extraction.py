import xml.etree.ElementTree as ET
import re


def parse_xml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    return root


def get_article_id(path: str):
    return path.split('/')[-1]


def get_body(root: ET.Element):
    return ' '.join([x.text for x in root.find('.//{*}block') if re.match(r"[^\s]+.*", x.text) != None])


def get_headline(root: ET.Element):
    return ''.join([x.text for x in root.find('.//{*}hedline')])


def create_article(path: str):
    root = parse_xml(path)
    return {'id': get_article_id(path), 'headline': get_headline(root), 'body': get_body(root)}


if __name__ == '__main__':
    article = create_article('/home/simba/Desktop/Data/2018-01-17/TabletXML/01_16_vendsyssel_ons_s016_11_nordjyll_1701_201801170000_noid1620180116225748021.xml')
    print(article)
