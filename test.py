import pathlib
import shutil

dirInput = str(input("Enter direction: "))
sortedDirInput = str(input("Enter direction for sorted images: "))

foundDir = pathlib.Path(dirInput)
foundSortedDir = pathlib.Path(sortedDirInput)

imageExtensions = ['.jpg', '.jpeg', 'png', 'webp']
imageNames = []
numbersOfImages=0

if foundDir.is_dir() and foundSortedDir.is_dir():
    for item in foundDir.iterdir():
        for imageExtensionsItem in imageExtensions:
            if item.name.endswith(imageExtensionsItem):
                imageNames.append(item.name)
else:
    print("Directory is not correct!")

for item in imageNames:
   shutil.move(dirInput + "/" + item, sortedDirInput + "/" + item)