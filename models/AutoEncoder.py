# model construction for AutoEncoder
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # define encoder
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512)
        )

        # define decoder
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,output_padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 16, 3, stride=2,output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 3, 3, stride=1), 
            nn.Sigmoid(),
        )

    # define forward process
    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder

