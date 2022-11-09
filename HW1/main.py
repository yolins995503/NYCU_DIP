import cv2
import numpy as np
import argparse
import random
import numpy as np
import skimage.util.noise as noise

def img_resize(image,resize_height, resize_width):

    image_shape=np.shape(image)
    height=image_shape[0]
    width=image_shape[1]
    if (resize_height is None) and (resize_width is None):
        return image
    if resize_height is None:
        resize_height=int(height*resize_width/width)
    elif resize_width is None:
        resize_width=int(width*resize_height/height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))

    return image

def RGB2GRAY(img):

    return cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

def sp_noise(image,prob):

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gasuss_noise(image, mean=0, var=0.001):
    
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)

    return out

def padding(image , mask_size):

    padding_size = mask_size // 2
    image_x = image.shape[0]
    image_y = image.shape[1]
    output_x = image_x + padding_size * 2
    output_y = image_y + padding_size * 2
    output = np.zeros ((output_x , output_y),dtype="uint8")

    for i in range(image_x):
        for j in range(image_y):
            output[i + padding_size][j + padding_size] = image[i][j] 

    return output

def zero_padding(image, padding_left,  padding_right , padding_top , padding_bottom):

    height = image.shape[0]
    width = image.shape[1]
    boundary_top = np.zeros((padding_top, width)) 
    boundary_bottom = np.zeros((padding_bottom, width)) 
    output = np.vstack((boundary_top, image, boundary_bottom))
    boundary_left = np.zeros((height + padding_top + padding_bottom, padding_left)) 
    boundary_right = np.zeros((height + padding_top + padding_bottom, padding_right)) 
    output = np.hstack((boundary_left, output, boundary_right))

    return output

def convolution(image , kernel):

    kernel_size = len(kernel)
    image_x  , image_y = image.shape
    output = np.zeros((image_x, image_y),dtype = "uint8")

    img_padding = padding(image , kernel_size)
    for i in range(image_x):
        for j in range(image_y):
            temp = []
            result = 0
            temp.append(img_padding[i : i + kernel_size , j : j + kernel_size ] )
            image_data = np.row_stack(temp).flatten()
            kernel_data = np.row_stack(kernel).flatten()
            
            for k in range(len(kernel_data)):
                result += image_data[k] * kernel_data[k]

            output[i][j] = result
    
    return output

def sharpen_filter(image , gray_scale=False):

    kernel = np.array([[-1, -1, -1] , [-1, 8, -1] , [-1, -1, -1]])
    if gray_scale:
        output = convolution(image,kernel)

    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = convolution(image_r , kernel)
        output_g = convolution(image_g , kernel)
        output_b = convolution(image_b , kernel)
        output = np.dstack((output_r, output_g, output_b))

    return output

def unsharp_masking(image, kernel_size, k):

    mask = image - box_filter(np.copy(image), kernel_size) 
    output = image + k*mask

    return output

def box_filter(image , gray_scale=False):

    kernel = np.array([[1/9, 1/9, 1/9] , [1/9, 1/9, 1/9] , [1/9, 1/9, 1/9]])
    if gray_scale:
        output = convolution(image,kernel)

    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = convolution(image_r , kernel)
        output_g = convolution(image_g , kernel)
        output_b = convolution(image_b , kernel)
        output = np.dstack((output_r, output_g, output_b))

    return output

def Laplacian_filter(image , gray_scale=False):

    kernel = np.array([[0, -1, 0] , [-1, 5, -1] , [0, -1, 0]])
    if gray_scale:
        output = convolution(image,kernel)

    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = convolution(image_r , kernel)
        output_g = convolution(image_g , kernel)
        output_b = convolution(image_b , kernel)
        output = np.dstack((output_r, output_g, output_b))

    return output

def medium_filter(image , kernel_size , gray_scale = False):

    def cal_medium(image,kernel_size ):
        image_x = image.shape[0]
        image_y = image.shape[1]
        output = np.zeros((image_x, image_y),dtype = "uint8")

        padding_image = padding(image , kernel_size)

        for i in range(image_x):
            for j in range(image_y):
                temp = []
                temp.append(padding_image[i : i + kernel_size , j : j + kernel_size])
                data = np.sort(np.row_stack(temp).flatten())
                output[i][j] = data[len(data) // 2]
                
        return output

    if gray_scale:
        output = cal_medium(image,kernel_size)

    else:
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = cal_medium(image_r , kernel_size )
        output_g = cal_medium(image_g , kernel_size )
        output_b = cal_medium(image_b , kernel_size )
        output = np.dstack((output_r, output_g, output_b))

    return output

def max_filter(image , kernel_size , gray_scale = False):

    def maxfilter(image , kernel_size):

        image = np.copy(image)
        height = image.shape[0]
        width = image.shape[1]
        kernel_h, kernel_w = (kernel_size-1)//2, (kernel_size-1)//2
        image = zero_padding(image, kernel_w, kernel_w, kernel_h, kernel_h)
        result = np.zeros(image.shape, dtype=np.uint8) 
        for i in range(kernel_h, height+kernel_h):
            for j in range(kernel_w, width+kernel_w):
                result[i, j] = np.max(image[i-kernel_h:i+kernel_h+1, j-kernel_w:j+kernel_w+1])
        result = result[kernel_h:height+kernel_h, kernel_w:width+kernel_w]

        return result

    if gray_scale:
        output = maxfilter(image,kernel_size)

    else:
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = maxfilter(image_r , kernel_size )
        output_g = maxfilter(image_g , kernel_size )
        output_b = maxfilter(image_b , kernel_size )
        output = np.dstack((output_r, output_g, output_b))

    return output

def min_filter(image , kernel_size , gray_scale = False):

    def minfilter(image , kernel_size):

        image = np.copy(image)
        height = image.shape[0]
        width = image.shape[1]
        kernel_h, kernel_w = (kernel_size-1)//2, (kernel_size-1)//2
        image = zero_padding(image, kernel_w, kernel_w, kernel_h, kernel_h)
        result = np.zeros(image.shape, dtype=np.uint8) 
        for i in range(kernel_h, height+kernel_h):
            for j in range(kernel_w, width+kernel_w):
                result[i, j] = np.min(image[i-kernel_h:i+kernel_h+1, j-kernel_w:j+kernel_w+1])
        result = result[kernel_h:height+kernel_h, kernel_w:width+kernel_w]

        return result

    if gray_scale:
        output = minfilter(image,kernel_size)

    else:
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = minfilter(image_r , kernel_size )
        output_g = minfilter(image_g , kernel_size )
        output_b = minfilter(image_b , kernel_size )
        output = np.dstack((output_r, output_g, output_b))

    return output

def Histogram_Equalization(image , gray_scale=False):

    def cal_hiseq(image):
        data = np.zeros(256).astype(np.int64)
        image = image.flatten()
        for i in image:
            data[i] += 1

        return data

    def hiseq(image):
        data = cal_hiseq(image)
        p = data/image.size
        p_sum = np.cumsum(p)
        equal = np.around(p_sum * 255).astype('uint8')

        return equal[image]

    if gray_scale:
        output = hiseq(image)

    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = hiseq(image_r )
        output_g = hiseq(image_g )
        output_b = hiseq(image_b )
        output = np.dstack((output_r, output_g, output_b))

    return output 

def Power_Law(image,value):

    output = np.array(255*(image/255)**value , dtype = 'uint8')

    return output            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-image1" , type = str , default="./picture/gull.png")
    parser.add_argument("-image2" , type = str , default="./picture/flying.png")
    parser.add_argument("-image3" , type = str , default="./picture/cameraman.png")
    parser.add_argument("-image4" , type = str , default="./picture/color_sunset.png")
    parser.add_argument("-image5" , type = str , default="./picture/iris.png")
    parser.add_argument("-image6" , type = str , default="./picture/stairs.png")
    parser.add_argument("-image7" , type = str , default="./picture/blocks.jpg")
    parser.add_argument("-image8" , type = str , default="./picture/flying-88484_1280.jpg")
    parser.add_argument("-image9" , type = str , default="./picture/street.png")


    args = parser.parse_args()


    # img_input = cv2.imread(args.image6)
    # img_hiseq = Histogram_Equalization(img_input , gray_scale = False)
    # cv2.imwrite("img_hiseq.png" , np.hstack([img_input , img_hiseq]))

    # img_input = cv2.imread(args.image3)
    # img_power1 = Power_Law(img_input, 3)
    # img_power2 = Power_Law(img_input, 5)
    # img_power3 = Power_Law(img_input, 7)
    # cv2.imwrite("img_power1.png" , np.hstack([img_input , img_power1 , img_power2 ,img_power3]))
    # img_power1 = Power_Law(img_input, 0.5)
    # img_power2 = Power_Law(img_input, 0.4)
    # img_power3 = Power_Law(img_input, 0.3)
    # cv2.imwrite("img_power2.png" , np.hstack([img_input , img_power1 , img_power2 ,img_power3]))

    # img_input = cv2.imread(args.image7)
    # img_input = RGB2GRAY(img_input)
    # img_max1= max_filter(img_input , 3 , gray_scale = True)
    # img_max2= max_filter(img_input , 5 , gray_scale = True)
    # img_max3= max_filter(img_input , 7 , gray_scale = True)
    # cv2.imwrite("img_max_filter.png" , np.hstack([img_input , img_max1 , img_max2 ,img_max3]))

    # img_input = cv2.imread(args.image7)
    # img_input = RGB2GRAY(img_input)
    # img_min1= min_filter(img_input , 3 , gray_scale = True)
    # img_min2= min_filter(img_input , 5 , gray_scale = True)
    # img_min3= min_filter(img_input , 7 , gray_scale = True)
    # cv2.imwrite("img_min_filter.png" , np.hstack([img_input , img_min1 , img_min2 ,img_min3]))


    # img_input = cv2.imread(args.image7)
    # img_input = RGB2GRAY(img_input)
    # img_med1= medium_filter(img_input , 3 , gray_scale = True)
    # img_med2= medium_filter(img_input , 5 , gray_scale = True)
    # img_med3= medium_filter(img_input , 7 , gray_scale = True)
    # cv2.imwrite("img_med_filter.png" , np.hstack([img_input , img_med1 , img_med2 ,img_med3]))

    # img_input = cv2.imread(args.image7)
    # img_input = RGB2GRAY(img_input)
    # img_box1= box_filter(img_input , gray_scale = True)
    # img_box2= box_filter(img_input , gray_scale = True)
    # img_box3= box_filter(img_input , gray_scale = True)
    # cv2.imwrite("img_box_filter.png" , np.hstack([img_input , img_box1 , img_box2 ,img_box3]))

    # img_input = cv2.imread(args.image1)
    # img_input = RGB2GRAY(img_input)
    # img_SP1 = sp_noise(img_input , 0.03)
    # img_SP2 = sp_noise(img_input , 0.05)
    # img_SP3 = sp_noise(img_input , 0.07)
    # cv2.imwrite("img_SP.png" , np.hstack([img_input , img_SP1 , img_SP2 ,img_SP3]))

    # img_max1= max_filter(img_SP1 , 3 , gray_scale = True)
    # img_max2= max_filter(img_SP2 , 5 , gray_scale = True)
    # img_max3= max_filter(img_SP3, 7 , gray_scale = True)
    # cv2.imwrite("img_max_filter_SP.png" , np.hstack([img_input , img_max1 , img_max2 ,img_max3]))
    # img_min1= min_filter(img_SP1 , 3 , gray_scale = True)
    # img_min2= min_filter(img_SP2 , 5 , gray_scale = True)
    # img_min3= min_filter(img_SP3 , 7 , gray_scale = True)
    # cv2.imwrite("img_min_filter_SP.png" , np.hstack([img_input , img_min1 , img_min2 ,img_min3]))
    # img_med1= medium_filter(img_SP1 , 3 , gray_scale = True)
    # img_med2= medium_filter(img_SP2 , 5 , gray_scale = True)
    # img_med3= medium_filter(img_SP3 , 7 , gray_scale = True)
    # cv2.imwrite("img_med_filter_SP.png" , np.hstack([img_input , img_med1 , img_med2 ,img_med3]))
    # img_box1= box_filter(img_input , gray_scale = True)
    # img_box2= box_filter(img_input , gray_scale = True)
    # img_box3= box_filter(img_input , gray_scale = True)
    # cv2.imwrite("img_box_filter_SP.png" , np.hstack([img_input , img_box1 , img_box2 ,img_box3]))

    # img_input = cv2.imread(args.image2)
    # img_input = RGB2GRAY(img_input)
    # img_GasNo1 = gasuss_noise(img_input , 0 , 0.01)
    # img_GasNo2 = gasuss_noise(img_input , 0 , 0.05)
    # img_GasNo3 = gasuss_noise(img_input , 0 , 0.07)
    # cv2.imwrite("img_GasNo.png" , np.hstack([img_input , img_GasNo1 , img_GasNo2 ,img_GasNo3]))

    # img_max1= max_filter(img_GasNo1 , 3 , gray_scale = True)
    # img_max2= max_filter(img_GasNo2 , 5 , gray_scale = True)
    # img_max3= max_filter(img_GasNo3, 7 , gray_scale = True)
    # cv2.imwrite("img_max_filter_GasNo.png" , np.hstack([img_input , img_max1 , img_max2 ,img_max3]))
    # img_min1= min_filter(img_GasNo1 , 3 , gray_scale = True)
    # img_min2= min_filter(img_GasNo2 , 5 , gray_scale = True)
    # img_min3= min_filter(img_GasNo3 , 7 , gray_scale = True)
    # cv2.imwrite("img_min_filter_GasNo.png" , np.hstack([img_input , img_min1 , img_min2 ,img_min3]))
    # img_med1= medium_filter(img_GasNo1 , 3 , gray_scale = True)
    # img_med2= medium_filter(img_GasNo2 , 5 , gray_scale = True)
    # img_med3= medium_filter(img_GasNo3 , 7 , gray_scale = True)
    # cv2.imwrite("img_med_filter_GasNo.png" , np.hstack([img_input , img_med1 , img_med2 ,img_med3]))
    # img_box1= box_filter(img_GasNo1 , gray_scale = True)
    # img_box2= box_filter(img_GasNo2, gray_scale = True)
    # img_box3= box_filter(img_GasNo3, gray_scale = True)
    # cv2.imwrite("img_box_filter_GasNO.png" , np.hstack([img_input , img_box1 , img_box2 ,img_box3]))

    # img_input = cv2.imread(args.image1)
    # img_input = RGB2GRAY(img_input)
    # img_Laplacian = Laplacian_filter(img_input , gray_scale = True)
    # cv2.imwrite("img_Laplacian_filter_1.png" , np.hstack([img_input , img_Laplacian ]))

    # img_input = cv2.imread(args.image1)
    # img_input = RGB2GRAY(img_input)
    # img_unsharp_masking = unsharp_masking(img_input , 3,1)
    # cv2.imwrite("img_unsharp_masking_1.png" , np.hstack([img_input , img_unsharp_masking ]))





    img_input = cv2.imread(args.image1)
    img_input = RGB2GRAY(img_input)
    img_SP1 = sp_noise(img_input , 0.03)
    img_SP2 = sp_noise(img_input , 0.05)
    img_SP3 = sp_noise(img_input , 0.07)
    #cv2.imwrite("img_SP.png" , np.hstack([img_input , img_SP1 , img_SP2 ,img_SP3]))

    img_max1= max_filter(img_SP1 , 3 , gray_scale = True)
    img_max2= max_filter(img_SP2 , 5 , gray_scale = True)
    img_max3= max_filter(img_SP3, 7 , gray_scale = True)
    #cv2.imwrite("img_max_filter_SP.png" , np.hstack([img_input , img_max1 , img_max2 ,img_max3]))
    img_min1= min_filter(img_SP1 , 3 , gray_scale = True)
    img_min2= min_filter(img_SP2 , 5 , gray_scale = True)
    img_min3= min_filter(img_SP3 , 7 , gray_scale = True)
    #cv2.imwrite("img_min_filter_SP.png" , np.hstack([img_input , img_min1 , img_min2 ,img_min3]))
    img_med1= medium_filter(img_SP1 , 3 , gray_scale = True)
    img_med2= medium_filter(img_SP2 , 5 , gray_scale = True)
    img_med3= medium_filter(img_SP3 , 7 , gray_scale = True)
    #cv2.imwrite("img_med_filter_SP.png" , np.hstack([img_input , img_med1 , img_med2 ,img_med3]))
    img_box1= box_filter(img_input , gray_scale = True)
    img_box2= box_filter(img_input , gray_scale = True)
    img_box3= box_filter(img_input , gray_scale = True)
    #cv2.imwrite("img_box_filter_SP.png" , np.hstack([img_input , img_box1 , img_box2 ,img_box3]))

    cv2.imwrite("Best_noise_reduction.png" , np.hstack([img_input , img_SP1 , img_med2]))
