#Day 01

赛题提供了训练集、验证集和测试集中所有字符的位置框

最先想到就是图片分类问题，但是如果不提供位置框呢？

简单入门思路：定长字符识别

专业字符识别思路：不定长字符识别

专业分类思路：检测再识别


```
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label 
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)
```
pytorch定义数据模块 

torch.utils.data.DataLoader 
是代表这一数据的抽象类，定义自己的数据类继承和重写这个抽象类，只需要定义__len__和__getitem__


