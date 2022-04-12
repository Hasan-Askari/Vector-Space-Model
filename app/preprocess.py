
for i in range(1, 449):
    doc = open('./Abstracts/' + str(i) + '.txt', 'r')
    tokens = ""
    for line in doc:
        tokens = tokens + line
        # print(line)

    