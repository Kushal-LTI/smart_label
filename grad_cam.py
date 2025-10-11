# grad_cam.py
import numpy as np
import torch
import torch.nn.functional as F
import cv2

def get_target_layer(model, model_name):
    """Gets the last convolutional block for Grad-CAM."""
    if 'resnet' in model_name.lower():
        return model.layer4[-1]
    elif 'mobile' in model_name.lower():
        return model.features[-1]
    return None

class GradCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.use_cuda = use_cuda
        if self.use_cuda: self.model = model.cuda()
        self.feature_maps, self.gradients = None, None
        target_layer.register_forward_hook(self._save_feature_maps)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output): self.feature_maps = output.detach()
    def _save_gradients(self, module, grad_input, grad_output): self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor):
        if self.use_cuda: input_tensor = input_tensor.cuda()
        output = self.model(input_tensor)
        self.model.zero_grad()
        torch.max(output).backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.feature_maps.size(1)):
            self.feature_maps[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(self.feature_maps, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

    def get_cam_image(self, input_tensor, original_image_np):
        heatmap = self(input_tensor)
        heatmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)
        superimposed_img = cv2.addWeighted(superimposed_img, 0.6, heatmap_colored, 0.4, 0)
        return superimposed_img