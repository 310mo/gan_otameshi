import torch
import torchvision.transforms as transforms
import os
from PIL import Image




root = 'face_data'
class CustomDataset(torch.utils.data.Dataset):
    classes = ['altria', 'jeanne', 'nero', 'okita']

    def __init__(self, root, transform=None):
        #指定する場合は前処理クラスを受け取る
        self.transform = transform
        #画像とラベルの一覧を保持するリスト
        self.images = []
        self.labels = []
  
        #画像を読み込むファイルパスを取得
        altria_path = os.path.join(root, 'altria')
        jeanne_path = os.path.join(root, 'jeanne')
        nero_path = os.path.join(root, 'nero')
        okita_path = os.path.join(root, 'okita')

        #画像の一覧を取得
        altria_images = os.listdir(altria_path)
        jeanne_images = os.listdir(jeanne_path)
        nero_images = os.listdir(nero_path)
        okita_images = os.listdir(okita_path)

        altria_labels = [0] * len(altria_images)
        jeanne_labels = [0] * len(jeanne_images)
        nero_labels = [0] * len(nero_images)
        okita_labels = [0] * len(okita_images)


        #1個のリストにする
        for image, label in zip(altria_images, altria_labels):
            self.images.append(os.path.join(altria_path, image))
            self.labels.append(label)
        for image, label in zip(jeanne_images, jeanne_labels):
            self.images.append(os.path.join(jeanne_path, image))
            self.labels.append(label)
        for image, label in zip(nero_images, nero_labels):
            self.images.append(os.path.join(nero_path, image))
            self.labels.append(label)
        for image, label in zip(okita_images, okita_labels):
            self.images.append(os.path.join(okita_path, image))
            self.labels.append(label)

    def __getitem__(self, index):
        #インデックスをもとに画像のファイルパスとラベルを取得
        image = self.images[index]
        label = self.labels[index]
        #画像ファイルパスから画像を読み込む
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        #前処理がある場合は前処理を入れる
        if self.transform is not None:
            image = self.transform(image)
        #画像とラベルのペアを返却
        return image, label

    def __len__(self):
        #データ数を指定
        return len(self.images)