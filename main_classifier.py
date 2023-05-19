# Description: main function for training image classifier
import argparse

def main(argv=None): 

    parser = argparse.ArgumentParser()

    # params for experimental description      
    parser.add_argument('--version', type=str, default="ImageClassifier_v1.0", help='experimental version')
    parser.add_argument('--description', type=str, default='training image classification model', help='experimental description')
    parser.add_argument('--script', type=str, default='Classifier', help='script for training')
    
    # params for data and log dir          
    parser.add_argument('--train_folder', type=str, default='./dataset/train_data', help='the training data path')
    parser.add_argument('--test_folder', type=str, default='./dataset/test_data', help='the testing data path')
    parser.add_argument('--log_dir', type=str, default='./logs', help='the log path')
    
    # params for training process
    parser.add_argument('--num_classes', type=int, default=3, help='the num of classes which your task should classify')
    parser.add_argument('--batch_size', type=int, default=4, help='the batch size')
    parser.add_argument('--image_size', type=int, default=100, help='the image size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='the learning rate')
    parser.add_argument('--max_epoch', type=int, default=100, help='the maximum epoch')
    parser.add_argument('--checkpoint', type=int, default=1000, help='the checkpoint')

    config = parser.parse_args()

    # call the training process
    package  = __import__('scripts.train_'+config.script, fromlist=True)
    Trainer  = getattr(package, 'Trainer')
    trainer = Trainer(config)
    trainer.train()
    
if __name__ == '__main__':
    main()
