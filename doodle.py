import numpy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2 as cv


classes = ["airplane", "ant", "banana", "baseball", "bird", "bucket", "butterfly", "cat", "coffee cup",
           "dolphin", "donut", "duck", "fish", "leaf", "mountain", "pencil", "smiley face", "snake", "umbrella",
           "wine bottle"]


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # input(channels) ,output, kernal sizes
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16*4*4 do conv1,2 and pool to get the last layer length
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


FILE = 'D:\\PycharmProjects\\real time doodle classifier\\doodle.pth'
state_dict = torch.load('D:\\PycharmProjects\\real time doodle classifier\\doodle.pth', map_location=torch.device('cpu'))

loaded_model = ConvNN()
loaded_model.load_state_dict(state_dict)
loaded_model.eval()

cnn_model = loaded_model

# drawing pad
drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None


# 255-white 0-black
# mouse callback function
def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=20)
            pt1_x, pt1_y = x, y
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=20)


img = np.zeros((420, 420, 1), np.uint8)
# pad = np.ones((28, 28))*255
cv.namedWindow('test draw')
cv.setMouseCallback('test draw', line_drawing)
guess=0
while 1:
    cv.imshow('test draw', img)
    resized_image = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
    running_img = resized_image
    x = torch.from_numpy(running_img)

    x = x.reshape(1, 1, 28, 28)

    test_out = cnn_model(x.float())
    p1 = guess
    _, guess = torch.max(test_out, 1)

    if p1 != guess:
        print("I think it's a ", classes[guess])

    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):  # press q to quit
        break
    if k == 27:  # escape key
        resized_image = cv.resize(img, (28, 28))
        img = resized_image
        x = torch.from_numpy(img)

        x = x.reshape(1, 1, 28, 28)

        test_out = cnn_model(x.float())
        _, predicted = torch.max(test_out, 1)

        print("It's a ", classes[guess])
        pad = np.ones((28, 28)) * 250
cv.imshow('Pad', img)
cv.destroyAllWindows()
