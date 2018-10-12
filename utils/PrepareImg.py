# record image path in .txt document:
def GeneImgTxt(typestr):
    assert typestr in('train','validation','test')
    f = open('./parts_'+typestr+'.txt','r')
    out = open('../datasets/'+typestr+'.txt','w')
    for line in f.readlines():
        folder, num = line.rstrip().split(' ')
        num = int(num)
        str_num=str(num)

        # add '0' before the number
        zeroLen = 4-len(str_num)
        zeros=''
        for i in range(zeroLen):
            zeros ='0'+zeros
        str_num = zeros+str_num
        file = folder+'_'+str_num
        # print('the string number is '+str_num)

        # print('the raw file is ',folder,
              #'\n the folder name is ',fname)
        dpath = '../datasets/lfw_funneled/'+folder+'/'+file+'.jpg'
        rpath = '../datasets/parts_lfw_funneled_gt_images/'+file+'.ppm'
        out.write(dpath+'\t'+rpath+'\n')
    out.close()
    f.close()




GeneImgTxt('train')
GeneImgTxt('validation')
GeneImgTxt('test')


