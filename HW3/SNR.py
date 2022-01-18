import cv2
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-img1",type =str ,default='/home/bsplab/Documents/yolin/DIP_HW3/image_resize_folder/balls.jpg')
    parser.add_argument("-img2",type =str ,default='/home/bsplab/Documents/yolin/DIP_HW3/image_resize_folder/basketball_decoder.jpeg')
    args = parser.parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    psnr = cv2.PSNR(img2, img1)
    print (psnr)