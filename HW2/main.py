import argparse
import cv2
import math
import numpy as np
import time
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt

def RGB2GRAY(img):

    return cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

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

def gaussian_filter(img , size = 5):

    kernel_size = size
    sigma = 1
    kernel_h = int(kernel_size / 2)
    kernel_w = int(kernel_size / 2)
    kernel = np.zeros((kernel_size,kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            s = (i-kernel_h)
            t = (j-kernel_w)
            kernel[i,j] = math.exp( -1 * ( s**2 + t**2 ) / (2*(sigma**2)) )  / (2*math.pi*(sigma**2))
    img = convolution(img , kernel)
    return img

def canny_edge_detection(img , mode , img_name):
    def deroGauss(w=5,s=1,angle=0):
        wlim = (w-1)/2
        y,x = np.meshgrid(np.arange(-wlim,wlim+1),np.arange(-wlim,wlim+1))
        G = np.exp(-np.sum((np.square(x),np.square(y)),axis=0)/(2*np.float64(s)**2))
        G = G/np.sum(G)
        dGdx = -np.multiply(x,G)/np.float64(s)**2
        dGdy = -np.multiply(y,G)/np.float64(s)**2
        angle = angle*math.pi/180
        dog = math.cos(angle)*dGdx + math.sin(angle)*dGdy
        return dog
    
    def bilatfilt(I,w,sd,sr):
        dim = I.shape
        Iout= np.zeros(dim)
        wlim = (w-1)//2
        y,x = np.meshgrid(np.arange(-wlim,wlim+1),np.arange(-wlim,wlim+1))
        g = np.exp(-np.sum((np.square(x),np.square(y)),axis=0)/(2*(np.float64(sd)**2)))
        Ipad = np.pad(I,(wlim,),'edge')
        for r in range(wlim,dim[0]+wlim):
            for c in range(wlim,dim[1]+wlim):
                Ix = Ipad[r-wlim:r+wlim+1,c-wlim:c+wlim+1]
                s = np.exp(-np.square(Ix-Ipad[r,c])/(2*(np.float64(sr)**2)))
                k = np.multiply(g,s)
                Iout[r-wlim,c-wlim] = np.sum(np.multiply(k,Ix))/np.sum(k)
        return Iout

    # define function
    # Use the graussian filter to denoise
    def get_edges(I,sd):
        dim = I.shape
        Idog2d = np.zeros((nang,dim[0],dim[1]))
        for i in range(nang):
            dog2d = deroGauss(5,sd,angles[i])
            Idog2dtemp = abs(conv2(I,dog2d,mode='same',boundary='fill'))
            Idog2dtemp[Idog2dtemp<0]=0
            Idog2d[i,:,:] = Idog2dtemp
        return Idog2d

    # compute the gradient
    def calc_sigt(I,threshval):
        M,N = I.shape
        ulim = np.uint8(np.max(I))	
        N1 = np.count_nonzero(I>threshval)
        N2 = np.count_nonzero(I<=threshval)
        w1 = np.float64(N1)/(M*N)
        w2 = np.float64(N2)/(M*N)
        try:
            u1 = sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N1 for i in range(threshval+1,ulim))
            u2 = sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N2 for i in range(threshval+1))
            
            uT = u1*w1+u2*w2
            sigt = w1*w2*(u1-u2)**2
        except:
            return 0
        return sigt

    # NMS
    def nonmaxsup(I,gradang):
        dim = I.shape
        Inms = np.zeros(dim)
        xshift = int(np.round(math.cos(gradang*np.pi/180)))
        yshift = int(np.round(math.sin(gradang*np.pi/180)))
        Ipad = np.pad(I,(1,),'constant',constant_values = (0,0))
        for r in range(1,dim[0]+1):
            for c in range(1,dim[1]+1):
                maggrad = [Ipad[r-xshift,c-yshift],Ipad[r,c],Ipad[r+xshift,c+yshift]]
                if Ipad[r,c] == np.max(maggrad):
                    Inms[r-1,c-1] = Ipad[r,c]
        return Inms

    # Threshold
    def threshold(I,uth):
        lth = uth/2.5
        Ith = np.zeros(I.shape)
        Ith[I>=uth] = 255
        Ith[I<lth] = 0
        Ith[np.multiply(I>=lth, I<uth)] = 100
        return Ith

    # hysteresis
    def hysteresis(I):
        r,c = I.shape
        Ipad = np.pad(I,(1,),'edge')
        c255 = np.count_nonzero(Ipad==255)
        imgchange = True
        for i in range(1,r+1):
            for j in range(1,c+1):
                if Ipad[i,j] == 100:
                    if np.count_nonzero(Ipad[r-1:r+1,c-1:c+1]==255)>0:
                        Ipad[i,j] = 255
                    else:
                        Ipad[i,j] = 0
        Ih = Ipad[1:r+1,1:c+1]
        return Ih

    # Obtain the best threshold
    def get_threshold(I):
        max_sigt = 0
        opt_t = 0
        ulim = np.uint8(np.max(I))
        print(ulim,)
        for t in range(ulim+1):
            sigt = calc_sigt(I,t)
            if sigt > max_sigt:
                max_sigt = sigt
                opt_t = t
        print ('optimal high threshold: ',opt_t,)
        return opt_t

    if mode == 'hand_craft':
        # Resize the image
        while img.shape[0] > 1100 or img.shape[1] > 1100:
            img = cv2.resize(img,None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
        # translate into gray scale
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #Obtain the size of image
        dim = img.shape
        # Start canny
        #Bilateral filtering
        print ('Bilateral filtering...')
        # bilatfilt
        gimg = bilatfilt(gimg,5,3,10)
        print ('after bilat: ',np.max(gimg))
        #Obtain the time
        stime = time.time()
        angles = [0,45,90,135]
        nang = len(angles)

        #Gradient of Image
        print ('Calculating Gradient...')
        img_edges = get_edges(gimg,2)
        print ('after gradient: ',np.max(img_edges))

        #Non-max suppression：（NMS）
        print ('Suppressing Non-maximas...')
        for n in range(nang):
            img_edges[n,:,:] = nonmaxsup(img_edges[n,:,:],angles[n])

        print ('after nms: ', np.max(img_edges),)

        img_edge = np.max(img_edges,axis=0)
        lim = np.uint8(np.max(img_edge))
        #plt.savefig('img_edge',dpi=300)

        # Compute the treshold
        print ('Calculating Threshold...')
        th = get_threshold(gimg)
        the = get_threshold(img_edge)

        # Obtain the best threshold
        print ('Thresholding...')
        img_edge = threshold(img_edge, the*0.25)

        print ('Applying Hysteresis...')
        img_edge = nonmaxsup(hysteresis(img_edge),90)

        img_canny = cv2.Canny(np.uint8(gimg),th/3,th)

        print( 'Time taken :: ', str(time.time()-stime)+' seconds...')

        cv2.imwrite(img_name+'3'+'.jpg', img_edge)
        cv2.imwrite(img_name+'4'+'.jpg', img_canny)

    if mode == 'tool_box':
        img_gray = RGB2GRAY(img)
        img_gaussian = gaussian_filter(img_gray , size = 7 )
        edges = cv2.Canny(img_gaussian,50,250)
        # plt.subplot(121),plt.imshow(img,cmap = 'gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        # plt.savefig("img_output_canny_tool_"+img_name+"_.png" , dpi=400)

        #cv2.imwrite(img_name+'1'+'.jpg', img)
        cv2.imwrite(img_name+'22'+'.jpg', edges)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-image1" , type = str , default="coins")
    args = parser.parse_args()

    img_input = cv2.imread('./pic4PR2/'+args.image1+'.jpg')
    canny_edge_detection(img_input , 'tool_box' , args.image1)
    #canny_edge_detection(img_input , 'hand_craft' , args.image1)
