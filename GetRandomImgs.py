import os, random, shutil

fileDir = r'F:\binlang\srcImgs\small\test-1029-S-3584'    #源图片文件夹路径
tarDir  = r'F:\binlang\srcImgs\small\test'    #移动到新的文件夹路径

#rate = 0.1
picknumber = 58

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

if __name__ == '__main__':
	moveFile(fileDir)















	
