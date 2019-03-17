from datetime import datetime
from formrecognition.imgprocessing import Imgprocessing



def main_program():
    new_file='example.jpg'
    img_proc=Imgprocessing(new_file)

    print()

if __name__ == '__main__':
    # ocr()
    s = datetime.now()
    main_program()
    e = datetime.now()
    print('程序运行时间', e - s)
