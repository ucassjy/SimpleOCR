
def give_dictionary():
    i = 0
    dictionary = {}
    with open('chinese.txt') as f:
        chara = f.readlines()
        for line in chara:
            word = line.split(':')[0]
            i += 1
            dictionary[word] = i
        i += 1
        dictionary[':'] = i
    return dictionary

# dictionary = give_dictionary()
# print(dictionary)
# #NOTE  "#" IS INCLUDE
# I FORGET METHOD DEAL WITH #
