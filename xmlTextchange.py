import os
from  lxml import etree as ET

for x in os.listdir("data/Annotations/"):
    with open("data/Annotations/{}".format(x), encoding='ISO-8859-1') as f:
        tree = ET.parse("data/Annotations/{}".format(x))
        root = tree.getroot()

        for elem in root.getiterator():
            try:
                elem.text = elem.text.replace('hand', 'wrist')
                elem.text = elem.text.replace('jpeg', 'jpg')
                elem.text = elem.text.replace('png', 'jpg')
            except AttributeError:
                pass

    tree = ET.ElementTree(root)
    tree.write("Ann/" + x, encoding='ISO-8859-1')
    # break
