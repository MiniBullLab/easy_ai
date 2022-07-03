import os
import os.path
import glob


def com_path(input_path):
    _, file_name_post = os.path.split(input_path)
    file_name, _ = os.path.splitext(file_name_post)
    return int(file_name.strip())


class DirProcess():

    def __init__(self):
        pass

    def getDirFiles(self, dataDir, filePost="*.*"):
        if os.path.isdir(dataDir):
            imagePathPattern = os.path.join(dataDir, filePost)
            for filePath in glob.iglob(imagePathPattern):
                yield filePath
            return
        else:
            return None

    def getFileData(self, dataFilePath):
        with open(dataFilePath, 'r') as file:
            for line in file:
                if line.strip():
                    yield line.strip()
        return

    def sort_path(self, path_list):
        data_list = sorted(path_list, key=com_path)
        return data_list
