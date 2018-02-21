import csv
import tensorflow as tf
def create_onehot(index,depth):
    list =[]
    for i in range(depth):
        if i==index:
            list.append('1')
        else:
            list.append('0')
    return list
csvfile = open('allAnnotations.csv')
reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
class_list =[]
for row in reader:
    fixedrow = row[0].split(';')
    class_list.append(fixedrow[1])
csvfile.close()
csvfile = open('allAnnotations.csv')
reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
newlist = list(set(class_list))
with open('tags.csv', 'w') as writing_csvfile:
    filewriter = csv.writer(writing_csvfile,lineterminator='\n',delimiter=' ')
    for row in reader:
        fixedrow = row[0].split(';')
        one_hotvector = create_onehot(newlist.index(fixedrow[1]),len(newlist))
        filewriter.writerow(one_hotvector)



