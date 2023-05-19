# Function: some useful functions
import os
import shutil
import torchvision.transforms as transforms

# copy code to code_dir
def code_transfer(source, target, names):
    for name in names:
        shutil.copyfile(os.path.join(source, name), 
                        os.path.join(target, name))

# transform for image
def test_trans(size):
    trans = transforms.Compose([            
                transforms.Resize((size,size)),
                transforms.ToTensor(),
            ])
    return trans

# cal accumulated mseloss
class Matric_MSE(object):
    def __init__(self):
        self.holder = 0
        self.count = 0

    def add(self, loss):
        self.holder += loss.detach().cpu().numpy()
        self.count += 1

    def result(self):
        return float(self.holder)/float(self.count)
    
    def reset(self):
        self.holder = 0
        self.count = 0

# cal accumulated precision
class Precision(object):
    def __init__(self):
        self.true_num = 0
        self.all = 0

    def add(self, pre_tensor, trg_tensor):
        pred = pre_tensor.detach().cpu().numpy().argmax(-1)
        trg = trg_tensor.squeeze().detach().cpu().numpy()
        right_num = (pred==trg).sum()
        self.true_num += right_num
        self.all += pre_tensor.shape[0]

    def result(self):
        return float(self.true_num)/float(self.all)
    
    def reset(self):
        self.true_num = 0
        self.all = 0

# cal accumulated CrossEntropyLoss
class Matric_CE(object):
    def __init__(self):
        self.holder = 0
        self.count = 0

    def add(self, loss):
        self.holder += loss.detach().cpu().numpy()
        self.count += 1

    def result(self):
        return float(self.holder)/float(self.count)
    
    def reset(self):
        self.holder = 0
        self.count = 0

