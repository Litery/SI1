import itertools as it

def fun1(x):
    return (x * 7 + 3) % 5


def generator1(x1):
    while(True):
        yield x1
        x1 = fun1(x1)

def compare(gen, list):
    correct = True
    while correct:
        for x in list[:]:
            nexts = next(gen)
            print('  ' + str(x) + ' = ' + str(nexts))
            if nexts is not x:
                correct = False
                break
    if correct:
        return len(list)
    else:
        return 0


def search(gen):
    lists = []
    mins = 0
    while mins is 0:
        nexts = next(gen)
        print(nexts)
        if len(lists) is not 0 and lists[0] == nexts:
            mins = compare(generator1(nexts), lists)
        lists.append(nexts)


gen1 = generator1(4)
search(gen1)
