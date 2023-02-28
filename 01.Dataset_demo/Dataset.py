import os 
from torch.utils.data import Dataset
from PIL import Image

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.imgs_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.imgs_path[index]
        img_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_path)
        label = self.label_dir
        # print(img_path) 如果想打印某张图片的路径，可以取消注释
        return img, label
    
    def __len__(self):
        return len(self.imgs_path)

root_dir = "01.Dataset_demo\\data\\train"
ants_dir = "ants"
bees_dir = "bees"
ants_dataset = MyData(root_dir, ants_dir)
bees_dataset = MyData(root_dir, bees_dir)
Dataset = ants_dataset + bees_dataset

print(len(ants_dataset))
print(len(bees_dataset))
print(len(Dataset))

#显示图片
img, label = Dataset[244]
img.show()
