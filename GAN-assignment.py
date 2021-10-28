''' Example code from canvas'''




''' Our Implementation'''
import torch
from PIL.Image import Image
import os

### Data loader ###
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transformers):
        self.categories =('Bread', 'Dairy', 'Dessert', 'Egg', 'Fried', 'Meat', 'Pasta', 'Rice','Seafood', 'Soup','Vegetable/Fruit')
        self.root = root
        self.transforms = transformers
        # load all image files, sorting them to
        # ensure that they are aligned
        self.classes = torch.tensor([0,1,2,3,4,5,6,7,8,9,10])
        self.imgs = []
        for index in range(11):
            newimgs = list(sorted(os.listdir("Data/food-11/"+root+"/"+str(index))))
            for i in newimgs:
                b = i.split('.')
                if (b[1] == "jpg") & (len(i) > 0):
                    self.imgs.append(["Data/food-11/"+root+"/"+str(index)+"/"+i,self.classes[index]])



    def __getitem__(self, idx):
        # load images and masks
        ImgPath,ImgClass = self.imgs[idx]
        AddRandomTimestamp(idx, Image.open(ImgPath))
        img = self.transforms(Image.open(ImgPath))
        return(img,ImgClass)



    def __len__(self):
        return len(self.imgs)

def AddRandomTimestamp(id, img):
    ### Add timestamp randomly ###
    image = img
    print(img)
    return image

if __name__ == '__main__':
    dataset = CustomDataset("training", None)
    dataset.__getitem__(5)
    print(dataset.categories)