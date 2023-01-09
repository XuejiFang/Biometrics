import os

path  = '../face_recognition_system/database/Me/'
name = 'Hakim'
for i, name in enumerate(os.listdir(path)):
    print(name)

    newName = name + str(i) + '.png'
    os.rename(path+name, path+newName)