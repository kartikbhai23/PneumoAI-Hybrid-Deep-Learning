"""PneumoAI DenseNet121 + BiLSTM architecture and Grad-CAM visualization."""

import base64
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models


class DenseNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # The trained checkpoint supplies the learned parameters.
        self.features = models.densenet121(weights=None).features

    def forward(self, x):
        x = torch.relu(self.features(x))
        return x.flatten(2).transpose(1, 2)


class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        return x.mean(dim=1)


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = DenseNetFeatureExtractor()
        self.bilstm = BiLSTMModel()
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        return self.fc(self.bilstm(self.cnn(x)))


def gradcam_overlay(
    model: HybridModel,
    image: Image.Image,
    tensor: torch.Tensor,
    class_idx: int,
) -> str:
    """Return a PNG data URL showing Grad-CAM for the chosen class."""
    activation = None

    def forward_hook(_module, _inputs, output):
        nonlocal activation
        activation = output
        output.retain_grad()

    hook = model.cnn.features.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor)
            logits[0, class_idx].backward()
            if activation is None or activation.grad is None:
                raise RuntimeError("Could not calculate Grad-CAM gradients")

            gradients = activation.grad.detach()
            features = activation.detach()
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=(224, 224), mode="bilinear", align_corners=False
            )
            cam = cam[0, 0].cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)

    base = np.asarray(image.resize((224, 224)).convert("RGB"), dtype=np.float32)
    heat = np.stack(
        [np.ones_like(cam) * 255, cam * 210, np.zeros_like(cam)], axis=-1
    )
    alpha = (0.58 * cam)[..., None]
    overlay = np.uint8(np.clip(base * (1 - alpha) + heat * alpha, 0, 255))

    output = io.BytesIO()
    Image.fromarray(overlay).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")