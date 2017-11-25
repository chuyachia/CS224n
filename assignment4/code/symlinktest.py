import os


src = os.path.normpath('C:/Users/Client/Desktop/NLPwithDeepLearning/assignment4/train')
#src = 'train'
dst = '/Users/Client/AppData/Local/Temp/cs224n-squad-train'
if __name__=="__main__":
    #os.makedirs(dst)
#print(os.path.exists(src))
# This creates a symbolic link on python in tmp directory
    os.symlink(src,dst)
    #os.unlink(dst)
    print("symlink created")
