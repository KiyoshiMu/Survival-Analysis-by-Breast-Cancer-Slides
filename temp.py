# from collections import defaultdict
with open('back.md', 'r', encoding='utf8') as base:
    with open('temp.md', 'w', encoding='utf8') as temp:
        counts = [0]
        pre = 0
        for line in base.readlines():
            if line.startswith('#'):
                for count, char in enumerate(line, start=-1):
                    if char != '#':
                        break
                if count > pre:
                    try:
                        counts[count] = 1
                    except IndexError:
                        counts.append(1)
                elif count == pre:
                    counts[count] += 1
                else:
                    while pre < count:
                        counts[pre] = 0
                        pre += 1
                    counts[count] += 1
                pre = count
                # print()
                    # sub = "".join(['sub'] * (count-4))
                    # line = f'\\{sub}section' + '{' +line[count:-1] + '}'
                line = line[:count+2] + '.'.join([str(num) for num in counts[1:pre+1]]) + line[count+1:]
            # elif line == '\n':
            #     pluss = True
            # elif pluss:
            #     line = '  ' + line
            #     pluss = False
            temp.write(line)