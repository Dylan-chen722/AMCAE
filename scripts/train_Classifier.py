# train image classification model
import datetime
import os
import torch
from Tools.logger import Logger
from Tools.utils import (Matric_CE, Precision, code_transfer,
                         test_trans)
from torchvision import datasets
from models.ImageClassifier import ImageClassifier

# deine training and testing process for image classication model
class Trainer(object):
    def __init__(self, config):
        # get the path
        self.save_dir = os.path.join(config.log_dir, config.version)
        self.code_dir = os.path.join(self.save_dir, 'codes')
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        # create the path
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # print the parameters
        timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d_%H-%M-%S')
        self.logger = Logger(os.path.join(self.save_dir, f'record_{timeStr}.txt'))
        self.logger.log_param(config)
        # get the train data and test data
        self.train_data = datasets.ImageFolder(config.train_folder, transform = test_trans(config.image_size))
        self.test_data = datasets.ImageFolder(config.test_folder, transform = test_trans(config.image_size))
        # copy code to code_dir
        code_transfer("./scripts", self.code_dir, [f'train_{config.script}.py'])
        code_transfer("./models", self.code_dir, ['ImageClassifier.py'])
        code_transfer("./Tools", self.code_dir, ['data_loader.py', 'logger.py', 'utils.py', "mask.py"])
        # load model
        self.model = ImageClassifier(num_classes = config.num_classes).cuda()
        self.config = config

    def train(self):
        # define optimizer and scheduler
        config = self.config
        optimizer1 = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=config.max_epoch)
        # define loss function
        criterion = torch.nn.CrossEntropyLoss()
        train_metric = Precision()
        loss_metric = Matric_CE()
        # training the image classification model
        iters = 0
        for epoch in range(config.max_epoch):
            # load the training dataset
            train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.config.batch_size, shuffle=True)
            # get image and label from data_loader
            for k,(image, label) in enumerate(train_data_loader):
                image, label = image.cuda(), label.cuda()
                # obtain the predicted class
                _, prediction = self.model(image)
                # calculate the loss and accuracy
                loss = criterion(prediction, label)
                train_metric.add(prediction, label)
                loss_metric.add(loss)
                # update the parameters
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                # save the model and print the result
                if iters%config.checkpoint==0:
                    record = self.logger.record(Iter=iters, 
                                                Loss=loss_metric.result(), 
                                                TrainAcc=train_metric.result(),
                                                TestAcc_and_loss=self.test(),
                                                Epoch=epoch)
                    print(record)
                    torch.save([self.model.state_dict()], 
                                   os.path.join(self.checkpoint_dir, f'iter_{iters}.pth'))
                    train_metric.reset()
                    loss_metric.reset()
                    self.model.train()
                    
                iters+=1
            # update the learning rate
            scheduler1.step()



            
    # testing process
    def test(self):
        # define and reset the loss matrix and the precision matrix
        self.model.eval()
        test_metric = Precision()
        test_metric.reset()
        loss_metric = Matric_CE()
        loss_metric.reset()
        # define loss function
        criterion = torch.nn.CrossEntropyLoss()
        # get the test dataset
        data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.config.batch_size, shuffle=True)
        with torch.no_grad():
            for j,(image, label) in enumerate(data_loader):                
                image, label = image.cuda(), label.cuda()
                # obtain the result of image classification
                _, prediction = self.model(image)
                # calculate the loss
                loss = criterion(prediction, label)
                loss_metric.add(loss)
                test_metric.add(prediction, label)
        return test_metric.result(),loss_metric.result()