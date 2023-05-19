# train CAE model
import os
import datetime
import torch
from models.AutoEncoder import AutoEncoder
import torchvision.transforms as transforms
import datetime
import numpy as np
from Tools.data_loader import Folders
from Tools.logger import Logger
from Tools.utils import code_transfer, Matric_MSE, test_trans
from Tools.mask import apply_random_mask

# deine training and testing process for CAE model
class Trainer(object):

    # initialize the model and parameters
    def __init__(self, config):
        # get the path
        self.save_dir = os.path.join(config.log_dir, config.version)
        self.code_dir = os.path.join(self.save_dir, 'codes')
        self.fig_dir = os.path.join(self.save_dir, 'fig')
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        # create the path
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # print the parameters
        timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d_%H-%M-%S')
        self.logger = Logger(os.path.join(self.save_dir, f'record_{timeStr}.txt'))
        self.logger.log_param(config)
        # get the train data and test data
        self.train_data = Folders(config.train_folder, transform=test_trans(config.image_size))
        self.test_data = Folders(config.test_folder, transform=test_trans(config.image_size))
        # copy code to code_dir
        code_transfer("./scripts", self.code_dir, [f'train_{config.script}.py'])
        code_transfer("./models", self.code_dir, ['AutoEncoder.py'])
        code_transfer("./Tools", self.code_dir, ['data_loader.py', 'logger.py', 'utils.py', "mask.py"])
        # load model
        self.model = AutoEncoder().cuda()
        self.config = config

    # define training process   
    def train(self):
        # define optimizer and scheduler
        config = self.config
        optimizer1 = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=config.max_epoch)
        # define loss function
        criterion = torch.nn.MSELoss()
        loss_metric = Matric_MSE()
        # training the CAE model
        iters = 0
        for epoch in range(config.max_epoch):
            # load the training dataset
            data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=config.batch_size, shuffle=True)
            # get image from data_loader
            for _,(image, v, f) in enumerate(data_loader):  
                # random mask for image
                masked_image = torch.from_numpy(np.array([np.array(apply_random_mask(image[i], config.image_size, config.mask_size, config.mask_ratio).detach().cpu().numpy()) for i in range(image.shape[0])]))
                image, masked_image = image.cuda(), masked_image.cuda()
                # obtain the result of image reconstruction
                _, prediction = self.model(masked_image)
                # calculate the loss
                loss = criterion(prediction, image)
                # record the loss
                loss_metric.add(loss)
                # update the parameters
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()              
                # save the model and print the result
                if iters%config.checkpoint==0:
                    combine_out = torch.cat([image[0], masked_image[0],prediction[0]], -1)
                    combine_img = transforms.ToPILImage()(combine_out)
                    combine_img.save(os.path.join(self.fig_dir, f'generated_fig_train_{iters}.jpg'))
                    record = self.logger.record(Iter=iters, 
                                                Train_Loss=loss_metric.result(), 
                                                Test_Loss=self.test(iters=iters),
                                                Epoch=epoch)
                    print(record)
                    torch.save([self.model.state_dict()], 
                                   os.path.join(self.checkpoint_dir, f'model_iter_{iters}.pth'))
                    torch.save([self.model.Encoder.state_dict()], 
                                   os.path.join(self.checkpoint_dir, f'encoder_iter_{iters}.pth'))
                    loss_metric.reset()
                    self.model.train()
                iters+=1
            # update the learning rate
            scheduler1.step()
    # testing process
    def test(self,iters):
        self.model.eval()       
        # define and reset the loss matric
        loss_metric = Matric_MSE()
        loss_metric.reset()
        # define loss function
        criterion = torch.nn.MSELoss()
        # get the test dataset
        data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.config.batch_size, shuffle=True)
        # test the model
        with torch.no_grad():
            k=0
            for j,(image, v, f) in enumerate(data_loader):                
                # random mask for image
                masked_image = torch.from_numpy(np.array([np.array(apply_random_mask(image[i], self.config.image_size, self.config.mask_size, self.config.mask_ratio).detach().cpu().numpy()) for i in range(image.shape[0])]))
                image, masked_image = image.cuda(), masked_image.cuda()
                # obtain the result of image reconstruction
                _, prediction = self.model(masked_image)
                # calculate the loss
                loss = criterion(prediction, image)
                loss_metric.add(loss)
            # save the generated image
            combine_out = torch.cat([image[0], masked_image[0],prediction[0]], -1)
            combine_img = transforms.ToPILImage()(combine_out)
            combine_img.save(os.path.join(self.fig_dir, f'generated_fig_test_{iters}.jpg'))
        return loss_metric.result()
