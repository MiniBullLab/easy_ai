import os
import json
from optparse import OptionParser


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program transform xml to json"

    parser.add_option("-i", "--input_path", dest="input_path",
                      type="string", default=None,
                      help="input path of dataset")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input path of dataset")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input_path' option is required to run this program")

    return options


def dir_count(dir_name):
    return len(os.listdir(dir_name))

def class_count(input_path):
    class_json_file = os.path.join(input_path, "class.json")
    class_file = open(class_json_file, 'r')
    classes_ = json.load(class_file)
    classes = [value for key, value in classes_.items()]

    class_counts = []
    for c in classes:
        image_path = os.path.join(input_path, "JPEGImages", str(c))
        class_num = dir_count(image_path)
        class_counts.append(class_num)

    return classes, class_counts

def main():
    print("process start...")
    options = parse_arguments()
    classes, class_counts = class_count(options.input_path)

    print("The distribution is: \n")
    for c in classes:
        cls_index = classes.index(c)
        print("{} num is {}".format(c, class_counts[cls_index]))
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()