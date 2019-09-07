import cv2
import sys
import os

#入力は python face_extract.py (dataディレクトリ以下の対象サブディレクトリ名) (face_dataディレクトリ以下に保存したいサブディレクトリ名)

face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

#ディレクトリがなかったら作成
if not os.path.exists('face_data'):
    os.mkdir('face_data')

mk_dir = os.path.join('face_data', sys.argv[2])
if not os.path.exists(mk_dir):
    os.mkdir(mk_dir)

#対象のディレクトリからデータを取得
target_dir = os.path.join('data', sys.argv[1])
file_list = os.listdir(target_dir)

for file in file_list:
    target_file = os.path.join(target_dir, file)
    print(target_file)
    img = cv2.imread(target_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = 0
    y = 0
    w = 0
    h = 0

    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        # 顔画像（グレースケール）
        roi_gray = gray[y:y+h, x:x+w]
        # 顔画像（カラースケール）
        roi_color = img[y:y+h, x:x+w]

    #画像を認識できていたら保存
    if h > 0 and w > 0:
        #リサイズ
        resize_img = cv2.resize(roi_color, (96, 96))

        cv2.imwrite(os.path.join(mk_dir, file), resize_img)