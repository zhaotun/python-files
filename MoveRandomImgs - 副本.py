import os, random, shutil

#将 dir 目录下 train 中的各个文件夹中的文件随机移动到 dir 创建的 val 同名目录


dir = r'D:\caffe-gpu\caffe-windows\examples\char_wt\RGB-LiveDetect-DataSets-1031'
rate = 0.2

if __name__ == '__main__':

      trainpath = dir + "\\train"
      for file in os.listdir(trainpath):  
         #print( file)
         filepath = os.path.join(trainpath,file)  #每个子文件夹
         #print("filepath = %s " %  filepath)

         files = os.listdir(filepath)         #每个子文件夹的所有文件列表
         filelength = len(files) 
         print("filelength = %d " % filelength)
         
         picklength = int(filelength * rate)
         #print("picklength = %d " % picklength)
         sample = random.sample(files, picklength) #从每个子文件夹中随机选取
         print("len-sample = %d " % len(sample))
         
         list = filepath.split("\\")
         valpath =  dir + "\\val\\" + str(list[len(list)-1])
         #print("valpath = %s"%valpath)
         isExists = os.path.exists(valpath)
         if not isExists:
            os.makedirs(valpath)           

         for name in sample: 
             #print("name = %s"%name)
             shutil.move(os.path.join(filepath, name), os.path.join(valpath, name))



