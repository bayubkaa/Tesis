import torch
from utils.create_dataset import get_transform
import cv2
from modules.model import ConvNet
from time import time
from config.read_config import config

config = config()['config']
classes = config['classes']
num_class = len(classes)
img_size = config['resize']

transform_img = get_transform(img_size=img_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(img): #input numpy img
    img = transform_img(img)
    img.unsqueeze_(0)
    img = img.to(device)
    model = ConvNet(num_class)
    model.load_state_dict(torch.load('last.pth'))
    model.eval()
    output = model(img)   
    confidence, predicted_index = torch.max(output, 1)
    return confidence.item(), predicted_index.item()

if __name__ == '__main__':
    starttime = time()
    img = cv2.imread('char_data/7/1-0-499.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    confidence, predicted_index = predict(img)
    print(f'predicted class: {classes[predicted_index]}')
    print(f'execution time: {time()-starttime}')



