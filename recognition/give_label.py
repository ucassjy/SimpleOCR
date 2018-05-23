import numpy as np

def give_label():
    nu = 0
    with open("dictionary.txt",'r') as f:
        dictionary = f.readlines()
        real_dict = {}
        for line in dictionary:
            key = line.split(':')[0]
            value = line.split(':')[1]
            value = value.split('\n')[0]
            real_dict[key] = value
        real_dict[':'] = len(real_dict)
    # NOW WE HAVE THE REAL DICTIONARY !!!
    final_label = []
    useful_filename = []
    MAX_CHARACTER = 50
    with open("target_label.txt",'r') as ff:
        label = ff.readlines()#[0:Num]

        for line in label:
            filename = line.split(',')[0]
            content = list(line.split(',')[1])

            if content[0] == "#":
                continue
            if len(content)==1:
                continue
            for element in range(len(content)):


                if content[element] == '\n':
                    content.remove('\n')
                    break


                content[element] = int(real_dict[content[element]])
            # rest = MAX_CHARACTER - len(content)
            # if rest < 0:
            #     rest = 0
            # stack = np.zeros(rest)
            # content = np.array(content) #TURN IT TO ARRAY FOR HSTACK
            # content = np.hstack((content, stack)).tolist() #HERE CONTENT IS LIST
            if (len(content)>MAX_CHARACTER):
                print("YOU ARE HALF DEAD!!!!!!!!!!!!!!!!!!!!")
                content = content[0:MAX_CHARACTER] #CUT
                                # MAYBE NEED CUT
            final_label.append(content)
            useful_filename.append(filename)

        return final_label, useful_filename
