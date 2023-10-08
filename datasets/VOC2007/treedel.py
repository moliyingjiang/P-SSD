#  批量移除xml标注中的某一个类别标签
import xml.etree.cElementTree as ET
import os


path_root = ['./Annotations']

CLASSES = ["blurred line"]   # 将含有tim的objec删掉

for anno_path in path_root:
    xml_list = os.listdir(anno_path)
    for axml in xml_list:
        path_xml = os.path.join(anno_path, axml)
        tree = ET.parse(path_xml)
        root = tree.getroot()
        print("root",root)

        for child in root.findall('object'):
            name = child.find('name').text
            print("name",name)
            if name in CLASSES:     # 这里可以反向写，不在Class的删掉
                root.remove(child)
        # 重写
        tree.write(os.path.join('./annotation', axml))  # 记得新建annotations_new文件