import os
from optparse import OptionParser


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program rename classify image name"

    parser.add_option("-i", "--inputPath", dest="inputPath",
                      type="string", default=None,
                      help="input path of dataset")

    (options, args) = parser.parse_args()

    if options.inputPath:
        if not os.path.exists(options.inputPath):
            parser.error("Could not find the input path of dataset")
        else:
            options.input_path = os.path.normpath(options.inputPath)
    else:
        parser.error("'inputPath' option is required to run this program")

    return options


def main():
    print("process start...")
    options = parse_arguments()
    index = 0
    for image_path in os.listdir(os.path.join(options.input_path)):
        print(image_path)
        temp_path, img_name = os.path.split(image_path)
        file_name, file_post = os.path.splitext(img_name)
        save_name = str(index) + file_post
        print(save_name)
        os.rename(image_path, os.path.join(options.input_path, save_name))

    print("End of game!!!")


if __name__ == "__main__":
    main()
