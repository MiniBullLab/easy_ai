import os
import xml
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def check_annotation(voc_data_dir, basename):
    annotations_dir = os.path.join(voc_data_dir, 'Annotations')
    xml_file = os.path.join(annotations_dir, f"{basename}.xml")
    if os.path.exists(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        found = False
        for child in root:
            if child.tag == 'object':
                found = True
        if not found:
            # object tag not found
            return False
    else:
        # xml file not found
        return False
    return True

def convert_trainval(voc_data_dir):
    main_dir = os.path.join(voc_data_dir, 'ImageSets', 'Main')
    lists = os.listdir(main_dir)
    print(f"lists: {lists}")

    train_file = os.path.join(voc_data_dir, 'ImageSets', 'Main', 'train.txt')
    train_samples = []
    for l in lists:
        if l.endswith("_train.txt"):
            l_train_file = os.path.join(main_dir, l)
            with open(l_train_file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    line = line.strip()
                    segment = line.split(" ")[0]
                    tag = line.split(" ")[-1].strip()
                    if tag == '1':
                        if segment.endswith(".jpg") or segment.endswith(".png"):
                            basename = os.path.splitext(os.path.basename(segment))[0]
                        else:
                            basename = os.path.basename(segment)
                        if check_annotation(voc_data_dir, basename):
                            train_samples.append(basename)
    print(f"train_samples={train_samples}")
    with open(train_file, 'w') as fp:
        fp.write('\n'.join(train_samples))
        fp.write('\n')

    val_file = os.path.join(voc_data_dir, 'ImageSets', 'Main', 'val.txt')
    val_samples = []
    for l in lists:
        if l.endswith("_val.txt"):
            l_val_file = os.path.join(main_dir, l)
            with open(l_val_file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    line = line.strip()
                    segment = line.split(" ")[0]
                    tag = line.split(" ")[-1].strip()
                    if tag == '1':
                        if segment.endswith(".jpg") or segment.endswith(".png"):
                            basename = os.path.splitext(os.path.basename(segment))[0]
                        else:
                            basename = os.path.basename(segment)
                        if check_annotation(voc_data_dir, basename):
                            val_samples.append(basename)
    print(f"val_samples={val_samples}")
    with open(val_file, 'w') as fp:
        fp.write('\n'.join(val_samples))
        fp.write('\n')
