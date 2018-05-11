i=0
dictionary = {}
def give_dictionary():
    with open('chinese.txt') as f:
        chara = f.readlines()
        for line in chara:
            word = line.split(':')[0]
            i += 1
            dictionary[word] = i
        i += 1
        dictionary[':'] = i
    return dictionary

#NOTE NO "#" NOT INCLUDE
