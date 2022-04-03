import jittor as jt
from jittor import init
from jittor import nn
from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer, find_layer, find_googlenet_layer, find_mobilenet_layer, find_shufflenet_layer, find_vit_layer

class BaseCAM(object):

    def __init__(self, model_dict, optimizer):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']
        self.optimizer = optimizer
        
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        
        #print(model_type, layer_name, self.model_arch)
        if ('vgg' in model_type.lower()):
            self.target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif ('resnet' in model_type.lower()):
            self.target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif ('densenet' in model_type.lower()):
            self.target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif ('alexnet' in model_type.lower()):
            self.target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif ('squeezenet' in model_type.lower()):
            self.target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif ('googlenet' in model_type.lower()):
            self.target_layer = find_googlenet_layer(self.model_arch, layer_name)
        elif ('shufflenet' in model_type.lower()):
            self.target_layer = find_shufflenet_layer(self.model_arch, layer_name)
        elif ('mobilenet' in model_type.lower()):
            self.target_layer = find_mobilenet_layer(self.model_arch, layer_name)
        elif ('vision_transformer' in model_type.lower()):
            self.target_layer = find_vit_layer(self.model_arch, layer_name)
        else:
            self.target_layer = find_layer(self.model_arch, layer_name)
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def execute(self, input, class_idx=None, retain_graph=False):
        return None

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)