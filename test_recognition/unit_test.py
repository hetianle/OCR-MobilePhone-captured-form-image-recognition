
import formrecognition.recogntion as re
import cv2

def test_seg():
    file="test_recognition/digit_image/50.png"

    img=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    list =re.ocr_grid_recognition(img)
    print()



test_seg()