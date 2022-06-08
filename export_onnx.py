import torch.onnx 

from modules.resnet import ResNet50

#Function to Convert to ONNX 
def Convert_ONNX(model, input_size): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "converted_model.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}
        ) 
    print(" ") 
    print('Model has been converted to ONNX')

if __name__ == "__main__":

    '''from char_classification.modules.mobilenetv2 import MobileNetV2
    model = MobileNetV2(ch_in=3, n_classes=36).to('cpu')
    model.load_state_dict(torch.load('char_classification\mbnetv2_myanmar_best.pth', map_location='cpu'))'''
    
    model = ResNet50(num_classes=2)
    #model = load_model('model/yolor/weights/character/best_overall.pt', 'model/yolor/cfg/yolor_character.cfg', 512, device='cpu')
    Convert_ONNX(model, (1,3,224,224))
