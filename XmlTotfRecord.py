import time
import os
import hashlib
import lxml.etree
import tqdm
import tensorflow as tf

'''_________________________________variable definition_________________________________'''
dataDir = "data/"
splitType = "val" #test and train
classesFile = "data/classes.names"
outputFile = "data/test.tfrecord"

'''_________________________________value prepration___________________________________'''
def valuePrepration(annotation, class_map):
    img_path = os.path.join(dataDir, 'JPEGImages', annotation['filename'])
    print(img_path)
    print(annotation['filename'])
    img_path = img_path.rsplit(".", 1)[0] + ".jpg"
    try:
        img_raw = open(img_path, 'rb').read()
    except:
        try:
            img_path = img_path.replace("wrist", "hand")
            img_raw = open(img_path, 'rb').read()
        except:
            try:
                img_path = img_path.replace("jpg", "png")
                img_path = img_path.rsplit(".", 1)[0] + ".jpg"
                img_raw = open(img_path, 'rb').read()
            except:
                img_path = img_path.replace("jpg", "jpeg")
                img_path = img_path.rsplit(".", 1)[0] + ".jpg"
                img_raw = open(img_path, 'rb').read()


    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin, ymin, xmax, ymax, classes, classes_text, truncated, views, difficult_obj = [], [], [], [], [],  [], [], [],  []
    if 'object' in annotation:
        for obj in annotation['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            print("========")
            print(obj['bndbox']['xmin'])
            print(obj['bndbox']['ymin'])
            print(obj['bndbox']['xmax'])
            print(obj['bndbox']['ymax'])
            print(class_map[obj['name']])
            print("========")

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def tfrConversion():
    class_map = {name: idx for idx, name in enumerate(open(classesFile).read().splitlines())}
    writer = tf.io.TFRecordWriter(outputFile)
    image_list = open(os.path.join(dataDir, 'ImageNameDetails', 'All_img_Deails%s.txt' % splitType)).read().splitlines()
    print(len(image_list))
    print (image_list[0].splitlines(), "======")
    for image in tqdm.tqdm(image_list):
        name, _ = image.rsplit(" -1", 1)
        annotation_xml = os.path.join(dataDir, 'Annotations', name + '.xml')
        annotation_xml = lxml.etree.fromstring(open(annotation_xml, "rb").read())
        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = valuePrepration(annotation, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("Done_______________")


if __name__ == '__main__':
    tfrConversion()
