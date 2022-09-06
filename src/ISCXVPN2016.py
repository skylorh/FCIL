from ast import main
from torchvision import datasets
from torchvision.datasets import CIFAR100
import numpy as np


class ISCXVPN2016():
    def __init__(self,
                root,
                train=True,
                transform=None,
                target_transform=None,
                test_transform=None,
                target_test_transform=None,
                ):

        self.dataset=datasets.ImageFolder(root)
        self.transform=transform
        self.target_transform=target_transform
        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.dataset[np.array(self.dataset.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels=self.concatenate(datas,labels)

    def getTrainData(self, classes, exemplar_set, exemplar_label_set):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]

        # 增加新类的数据
        for label in classes:
            data=self.dataset[np.array(self.dataset.targets)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getSampleData(self, classes, exemplar_set, exemplar_label_set, group):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]

        if group == 0:
            for label in classes:
                data=self.dataset[np.array(self.datasetset.targets)==label]
                datas.append(data)
                labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getTrainItem(self,index):
        img, target = self.TrainData[index], self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index,img,target

    def getTestItem(self,index):
        img, target = self.TestData[index], self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)


    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)

    def get_image_class(self,label):
        return self.dataset[np.array(self.dataset.targets)==label]



# def main() -> int:
#     train_datasets = ISCXVPN2016("./img_data", True)
#     return 0

# if __name__ == '__main__':
#     main()
