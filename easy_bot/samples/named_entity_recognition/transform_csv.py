import os
from operator import itemgetter

def transform_csv(csvfile):
    csvlines = []

    with open(csvfile, 'rt') as fp:
        lines = fp.readlines()
        csvlines = [l.strip().split(",") for l in lines]

    changeset = []
    files = {}
    for ridx, row in enumerate(csvlines):
        if ridx == 0:
            continue

        out = ""
        # print(row)
        
        docfile, idx, begin, end, tag = row
        idx, begin, end = int(idx), int(begin), int(end)
        row = (docfile, idx, begin, end, tag)

        if docfile not in files:
            doclines = []
            with open(docfile, 'rt') as fp:
                doclines = fp.readlines()
                doclines = [l.strip() for l in doclines]
            files[docfile] = doclines

        if len(changeset) > 0:
            if changeset[-1][0] == docfile and changeset[-1][1] == idx:
                changeset.append(row)
                continue
            else:
                # DON'T APPEND TO CHANGE-SET IMMEDIATELY, PERFORM A OVERWRITE INSTEAD
                files = overwrite(files, changeset)
                changeset = []
        else:
            changeset.append(row)
            continue
        
        # if ridx > 10:
        #     break

def overwrite(files, changeset):
    # print(changeset)
    changeset = sorted(changeset, key=itemgetter(2), reverse=False)
    # print(changeset)
    offset = 0
    docfile, idx, begin, end, tag = changeset[0]
    line = files[docfile][idx]

    for row in changeset:
        docfile, idx, begin, end, tag = row
        begin += offset
        end += offset
        newline = line[:begin] + "{{" + tag + ":" + line[begin:end] + "}}" + line[end:]
        offset += 5+len(tag)
        line = newline

    print(newline)
    # print()
    files[docfile][idx] = newline

    return files

transform_csv("data.csv")
