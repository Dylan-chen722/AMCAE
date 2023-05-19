# model construction for ImageClassifier
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.num_classes = num_classes
        # define feature extrator (Encoder)
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
            nn.BatchNorm2d(512),
            nn.AvgPool2d(6,1)
        )

        # define classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 1000),
            nn.ReLU(True),
            nn.Linear(1000, self.num_classes),
            nn.Softmax(dim = 1)
        )
    # define forward process
    def forward(self, x):
        x = self.Encoder(x)
        b = x.shape[0]
        y = x.reshape(b,-1)
        y = self.classifier(y)
        return x, y

