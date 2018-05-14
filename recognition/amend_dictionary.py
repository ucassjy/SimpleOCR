import numpy as np
from dictionary import give_dictionary

#NOTE: DESIGN FOR TRAINING DATA

def amend_dictionary():
    dictionary = give_dictionary()

    MAX_CHARACTER = 50

    with open("target_label.txt",'r') as f:
        label = f.readlines()#[0:Num]
        for line in label:
            filename = line.split(',')[0]
            content = list(line.split(',')[1])
            if content[0] == "#":
                continue
            for element in range(len(content)):


                if content[element] not in dictionary.keys() and element != len(content) -1:
                    dictionary[content[element]] = len(dictionary)

            #        print(len(dictionary))
        with open("dictionary.txt",'w') as ff:
            new_dictionary = {v:k for k,v in dictionary.items()}
            for i in range(1,len(new_dictionary)+1):
            #    print(len(new_dictionary))
            #    print(new_dictionary[i])
                ff.write(new_dictionary[i] + ':' + str(i) + '\n')

        ff.close()
amend_dictionary()




                    # if content[element] == "0":
                    #     content[element] = "\ufeff0"        #FUUUUUUUUUUCK! FUCKFUCKFUCKFUCKFUCKFUCKFUCK
                    # if element==len(content)-1:
                    #     content[element] = 0.0
        #         else:
        #             content[element] = dictionary[content[element]]
        #     rest = MAX_CHARACTER - len(content)
        #     if rest < 0:
        #         rest = 0
        #     stack = np.zeros(rest)
        #     content = np.array(content) #TURN IT TO ARRAY FOR HSTACK
        #     content = np.hstack((content, stack)).tolist() #HERE CONTENT IS LIST
        #     if (len(content)>MAX_CHARACTER):
        #         print("YOU ARE HALF DEAD!!!!!!!!!!!!!!!!!!!!")
        #         content = content[0:MAX_CHARACTER] #CUT
        #
        #         # MAYBE NEED CUT
        #     final_label.append(content)
        #
        #     useful_filename.append(filename)
        #
        # final_label = np.array(final_label)
        #
        # return final_label, useful_filename



# print(final_label.shape)
# print(len(useful_filename))

# #->0

# <#FILE> is not include.
