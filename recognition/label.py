import numpy as np
from dictionary import give_dictionary

def give_label(Num):
    dictionary = give_dictionary()
    final_label = []
    useful_filename = []
    MAX_CHARACTER = 50
    with open("target_label.txt",'r') as f:
        label = f.readlines()[0:Num]
        for line in label:
            filename = line.split(',')[0]
            content = list(line.split(',')[1])
            if content[0] == "#":
                continue
            for element in range(len(content)):
                if content[element] == "0":
                    content[element] = "\ufeff0"        #FUUUUUUUUUUCK! FUCKFUCKFUCKFUCKFUCKFUCKFUCK
                if element==len(content)-1:
                    content[element] = 0.0
                else:
                    content[element] = dictionary[content[element]]
            rest = MAX_CHARACTER - len(content)
            if rest < 0:
                rest = 0
            stack = np.zeros(rest)
            content = np.array(content) #TURN IT TO ARRAY FOR HSTACK
            content = np.hstack((content, stack)).tolist() #HERE CONTENT IS LIST
            if (len(content)>MAX_CHARACTER):
                print("YOU ARE DEAD!!!!!!!!!!!!!!!!!!!!")
                # MAYBE NEED CUT
            final_label.append(content)
            
            useful_filename.append(filename)
        
        final_label = np.array(final_label)
        
        return final_label, useful_filename

final_label, useful_filename = give_label(40)
print(final_label.shape)
print(len(useful_filename))

# #->0

# <#FILE> is not include.