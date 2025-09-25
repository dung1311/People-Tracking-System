import torch
import onnxruntime

class Network:
    def __init__(self, model_path, device):
         # Initialize model
        _providers = [('CUDAExecutionProvider', {"device_id": device})]
        self.session = onnxruntime.InferenceSession(model_path,
            providers=_providers)
        # Get model info
        self.get_input_details()
        self.get_output_details()
        
    
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
    
    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.output_shape = model_outputs[0].shape

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs