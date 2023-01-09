import os

path  = '../WSB2022-assignment/face_recognition_system/database/mei/'

for i, name in enumerate(os.listdir(path)):
    print(name)

    newName = 'KUN_' + str(i) + '.png'
    os.rename(path+name, path+newName)