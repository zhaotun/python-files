import os, random, shutil
import cv2


#源图片文件夹路径
source_path = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\images_1483'
mask_path   = r'F:\binlang\srcImgs\small\test_png'

labeled_path  = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\annotations_1483\trimaps'

#目标图片文件夹路径
target_path  = r'F:\binlang\srcImgs\small\test'    

#rate = 0.1
picknumber = 58

if os.path.exists(source_path):
    for root, dirs, files in os.walk(source_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
        for file in files:                          # files 是所有的文件名列表，
            #if file.endswith('_jpeg.png') or file.endswith('_jpeg_scoremap.png'):
            maskname = file.replace('.','_')
            maskname = maskname + ".png"
            maskimg = os.path.join(mask_path, maskname)
            print(maskimg)
            mask_img = cv2.imread(maskimg)
            #mask_resize_img = mask_img
            #cv2.imshow("before resize",mask_img)

            srcimgname = os.path.join(root, file)
            src_img    = cv2.imread(srcimgname)
            cv2.imshow("src_img",src_img)
            print(srcimgname)

            x, y = src_img.shape[0:2]
            mask_resize_img = cv2.resize(mask_img,(y,x), 0, 0, cv2.INTER_NEAREST)
            #cv2.imshow("after resize",mask_resize_img)
            
            labeled_imgname = os.path.join(labeled_path, file.replace('jpeg','png') )
            cv2.imwrite( labeled_imgname,mask_resize_img)

            print(mask_resize_img.shape) 
            height   = mask_resize_img.shape[0]
            weight   = mask_resize_img.shape[1]
            channels = mask_resize_img.shape[2]
            print("weight : %s, height : %s, channel : %s" %(weight, height, channels))
            
            for row in range(height):            #遍历高
               for col in range(weight):         #遍历宽
                  #for c in range(channels):     #便利通道
                     pv = mask_resize_img[row, col, 0] 
                     #print("pv is %d",pv)
                     if pv == 1: 
                        mask_resize_img[row, col, 0] = 255     #全部像素取反，实现一个反向效果
                        #mask_resize_img[row, col, c] = 255     #全部像素取反，实现一个反向效果
                        #mask_resize_img[row, col, c] = 255     #全部像素取反，实现一个反向效果

            cv2.imshow("fanxiang", mask_resize_img)
            cv2.waitKey(0)
            #src_file = os.path.join(root, file)
            #shutil.move(src_file, target_path)
            #print(src_file)
            #i=i+1
 
def moveFile(fileDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=len(pathDir)
        #p=rate    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        #picknumber=int(filenumber*p) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        print (sample)
        for name in sample:
            #shutil.move(fileDir+name, tarDir+name)
            shutil.move(os.path.join(fileDir,name) , os.path.join(tarDir,name))
  
        return

#if __name__ == '__main__':
#	moveFile(fileDir)
#	cv.waitkey(0)















	
