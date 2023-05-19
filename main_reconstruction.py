# Description: main function for obtaining the reconstructed image and residual image
import os
import torch
import torchvision.transforms as transforms
from models.AutoEncoder import AutoEncoder
from Tools.data_loader import Folders
from Tools.utils import test_trans
from Tools.mask import apply_region_mask
import PIL.ImageChops as IC

# load model 
model = AutoEncoder().cuda()
model.load_state_dict(torch.load("model_path.pth")[0])
model.eval()
# get reconstructed image and residual image
def image_reconstruction():
    # load data
    file_path = "./dataset"
    train_datasets = Folders(file_path, transform=test_trans(100))
    train_loader=torch.utils.data.DataLoader(train_datasets, batch_size=1, shuffle=False)
    # save dir
    save_path = "./results"
    # get reconstructed image
    for i, (image, label, filename) in enumerate(train_loader):
        with torch.no_grad():   
            image = image.cuda()
            # apply regional mask for image
            x1, x2, y1, y2 = 20, 80, 0, 70
            image_mask = apply_region_mask(image[0],x1,x2,y1,y2)
            image_mask = image_mask.unsqueeze(0)
            # obtain the result of image reconstruction           
            _, y = model (image_mask)
            y = y.squeeze(0)
            # save the reconstructed image
            combine_out = torch.cat([image[0],image_mask[0], y], -1)
            des_path = os.path.join(save_path,label[0],filename[0])
            combine_out = transforms.ToPILImage()(combine_out)
            combine_out.save(des_path)
            # save the residual image
            original_data = transforms.ToPILImage()(image[0])
            reconstructed_image = transforms.ToPILImage()(y)
            IC.subtract(original_data,reconstructed_image).save(os.path.join(save_path,label[0],"residual_"+filename[0]))

if __name__ == '__main__':
    image_reconstruction()


