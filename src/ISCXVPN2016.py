from torchvision import datasets
import numpy as np
from PIL import Image


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
            indexs_bool = np.array(self.dataset.targets)==label
            indexs = [idx for (idx,x) in enumerate(indexs_bool) if x]
            data=np.array([np.array(self.dataset[idx][0]) for idx in indexs])
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
            # print("targets label: ", np.array(self.dataset.targets), label)
            indexs_bool = np.array(self.dataset.targets)==label
            indexs = [idx for (idx,x) in enumerate(indexs_bool) if x]
            # print("indexs_bool: ", indexs_bool)
            # print("indexs: ", indexs)
            data=np.array([np.array(self.dataset[idx][0]) for idx in indexs])
            # print("data: ", data)
            # print("type data: ", type(data))
            # print("data.shape: ", data.shape)
            # print("label: ", np.full((data.shape[0]),label))
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
                indexs_bool = np.array(self.dataset.targets)==label
                indexs = [idx for (idx,x) in enumerate(indexs_bool) if x]
                data=np.array([np.array(self.dataset[idx][0]) for idx in indexs])
                datas.append(data)
                labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)


        # print("getTrainItem: ", img.shape)
        return index,img,target

    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

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
        indexs_bool = np.array(self.dataset.targets)==label
        indexs = [idx for (idx,x) in enumerate(indexs_bool) if x]
        data=np.array([np.array(self.dataset[idx][0]) for idx in indexs])
        return data
