import cv2
import os 
def resize_images(input_folder, out_folder ,img_name , height ,width):
    input_path = input_folder+'/'+img_name
    image = cv2.imread(input_path)
    resize_image = cv2.resize(image, (height, width), interpolation=cv2.INTER_AREA)
    out_path = out_folder+'/'+img_name
    cv2.imwrite( out_path, resize_image )

if __name__=='__main__':
    input_folder = 'pic4PR2'
    out_folder = 'image_resize_folder'
    files= os.listdir(input_folder)
    for idx in range(len(files)):
        resize_images( input_folder ,out_folder , files[idx],512,512)