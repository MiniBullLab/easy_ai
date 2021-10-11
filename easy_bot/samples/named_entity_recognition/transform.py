import random
random.seed(111)


def bosonnlp_to_bio2(origfile, trainfile, valfile):
    """annotations.csv
    File, Line, Begin Offset, End Offset, Type
    documents.txt, 0, 0, 11, ENGINEER
    documents.txt, 1, 0, 5, ENGINEER
    documents.txt, 3, 25, 30, MANAGER 
    """

    val_ratio = 0.2

    traindata = []
    valdata = []
    with open(origfile, 'rt') as fp:
        lines = fp.readlines()
        random.shuffle(lines)

        val_samples = int(len(lines) * val_ratio)
        val_lines = lines[:val_samples]
        train_lines = lines[val_samples:]

        def transform(line):
            # print(line)

            it = 0
            document = ""
            annotations = []
            while True:
                start = line.find("{{", it)
                end = line.find("}}", start)
                next_start = line.find("{{", end)
                
                if end < 0:
                    break

                # print(start, end)
                labeltext = line[start+2:end]
                loc = labeltext.find(":")
                label = labeltext[:loc]
                text = labeltext[loc+1:]

                # label, text = line[start+2:end].split(":")
                prefix = line[it:start]
                if next_start > 0:
                    suffix = line[end+2:next_start]
                else:
                    suffix = line[end+2:]
                
                tic = len(prefix) + len(document)
                toc = len(prefix) + len(document) + len(text)

                annotations.append([tic, toc, label])
                document += prefix + text + suffix

                it = next_start
            document = document.replace(' ', '_')
            document = document.replace('，', ',')
            document = document.replace('“', '"')
            document = document.replace('”', '"')
            document = document.replace('：', ':')
            document = document.replace('（', '(')
            document = document.replace('）', ')')
            document = document.replace('\t', '_')
            return annotations, document

        documents = []
        def strip_suffix(tag):
            # if tag.lower().endswith("name"):
            #     tag = tag[:-4]
            return tag.upper().strip("-_")
        with open(trainfile, 'w') as fp:
            for idx, line in enumerate(train_lines):
                annotations, document = transform(line)

                document = document.strip()
                documents.append(document)

                count = 0
                word = ""
                for i, c in enumerate(document):
                    if c.isalpha():
                        if len(document) > i+1:
                            if document[i+1].isalpha():
                                word += c
                                continue
                            else:
                                word += c
                                c = word
                                word = ""
                    elif c == "_":
                        continue

                    label = "O"
                    for a in annotations:
                        if i >= a[0] and i < a[1]:
                            label = "I-"+strip_suffix(a[2])
                    fp.write("{} X X {}\n".format(c, label))

                    # limit sequence length to 128 - 2
                    if (count % 125) == 124 or c in ["。", "；"]:
                        fp.write("\n")
                        count = 0
                    else:
                        count += 1

                fp.write("\n")

        with open(valfile, 'w') as fp:
            for idx, line in enumerate(val_lines):
                annotations, document = transform(line)

                # print(annotations)
                # print(document)
                document = document.strip()
                documents.append(document)

                count = 0
                word = ""
                for i, c in enumerate(document):
                    if c.isalpha():
                        if len(document) > i+1:
                            if document[i+1].isalpha():
                                word += c
                                continue
                            else:
                                word += c
                                c = word
                                word = ""
                    elif c == "_":
                        continue

                    label = "O"
                    for a in annotations:
                        if i >= a[0] and i < a[1]:
                            label = "I-"+strip_suffix(a[2])
                    fp.write("{} X X {}\n".format(c, label))

                    # limit sequence length to 128 - 2
                    if (count % 125) == 124 or c in ["。", "；"]:
                        fp.write("\n")
                        count = 0
                    else:
                        count += 1

                fp.write("\n")

        # with open("documents.txt", "w") as fp:
        #     fp.write("\n".join(documents))

# bosonnlp_to_bio2('yellowpage.txt', 'yellowpage_bio2_train.txt', 'yellowpage_bio2_val.txt')
# bosonnlp_to_bio2('yellowpage_v2.txt', 'yellowpage_bio2_train_v2.txt', 'yellowpage_bio2_val_v2.txt')
# bosonnlp_to_bio2('yellowpage_v3.txt', 'yellowpage_bio2_train_v3.txt', 'yellowpage_bio2_val_v3.txt')
bosonnlp_to_bio2('ratings.txt', 'ratings_bio2_train_v3.txt', 'ratings_bio2_val_v3.txt')
