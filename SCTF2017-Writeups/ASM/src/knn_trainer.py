import os
import cv2
import numpy as np
file_names = list(range(0, 13)) #0~9와 +,-,*
train = []
train_labels = []

for file_name in file_names:
    #각각의 파일 읽어오기
    path = './training_data/' + str(file_name) + '/'
    #각각의 이미지 갯수 카운
    file_count = len(next(os.walk(path))[2])
    for i in range(1, file_count + 1):
        img = cv2.imread(path + str(i) + '.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train.append(gray)
        train_labels.append(file_name)

x = np.array(train)
#20x20을 400으로 reshape하기기
train = x[:, :].reshape(-1, 400).astype(np.float32)
#숫자정보가 train_labels에 담기
train_labels = np.array(train_labels)[:, np.newaxis]

np.savez("trained.npz", train=train, train_labels=train_labels)