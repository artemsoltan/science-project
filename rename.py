import pathlib
import os

dirInput=str(input("Enter library: "))

foundDir = pathlib.Path(dirInput)

if foundDir.is_dir():
    for item in foundDir.iterdir():
        newFileName = str(item).replace('.jpeg','.jpg')
        old_file = os.path.join(dirInput, item)
        new_file = os.path.join(dirInput, newFileName)
        os.rename(old_file, new_file)
    print("Successful renamed!")
else:
    print('Dir is incorrect!')