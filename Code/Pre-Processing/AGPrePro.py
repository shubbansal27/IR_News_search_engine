path = 'D:\\Mohit IR\\DataSets\\AG'
classIde = ["Business","SciTech","Sports","World"]


num = 0
fp = open(path+'\\Test.csv','r')
for line in fp.readlines():
    splitLine = line.split(',',1)
    num = num+1
    classInd = classIde[(int(splitLine[0][1]))-1]
    outPath = path + '\\Test\\' + classInd + '\\'+str(num)+'.txt'
    fop = open(outPath,'w')
    fop.write(splitLine[1])
    fop.close()
fp.close()
    
    
    