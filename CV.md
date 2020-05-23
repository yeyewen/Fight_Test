# Datawhale 组队学习

## Day 01

### 定义读取数据集，定义读取数据dataloader
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

定义需要的数据类，可以通过迭代的方法来取得每一个数据，但是很难实现取batch 和shuffle或者多线程

去读取数据，所以Pytorch中提供了torch.utils.data.DataLoader 定义一个新的迭代器

```
train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), 
    batch_size=40, 
    shuffle=True, 
    num_workers=10,
)

```
### 定义分类模型

在pytorch里面编写神经网络，所有层结构和神经网络都来自于torch.nn,所有的模型构建都是从这个基类nn.Module继承的

于是有一下模板
```
class net_name(nn.Module):
    def __init__(self):
        super(net_name, self).__init__()
		self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size)
	
	def forward(self,x):
		s=self.conv1(x)
		return x
```
```
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
                
        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5
```
## Day 02  5/23
该baseline思路定义为一个定长字符模型,但是这样容易的导致模型精度低
模型搭建 模型损失函数定义部分 
``` 
model = SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
```

模型损失函数定义部分


提交baseline结果，线上0.31，增大迭代轮数，线上结果0.38

思路尝试修改batchsize和目标增强方式

采用思路二的方法

