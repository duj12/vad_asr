import os

import openvino.runtime
import torch

CURFILE_DIR = os.path.dirname(__file__)
CRNN_FEATURES_ONNX_PATH = os.path.join(CURFILE_DIR, "crnn.features.onnx")


class Features(torch.nn.Module):

    def __init__(self, device="cpu") -> None:
        self.device = device
        ie = openvino.runtime.Core()
        model_onnx = ie.read_model(model=CRNN_FEATURES_ONNX_PATH)
        self.vino_model = ie.compile_model(model=model_onnx,
                                           device_name=device.upper())
        self.output_layer = self.vino_model.output(0)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """推理接口.

            input_data:
                torch.rand([1, 1, 60, 64], dtype=torch.float32, device="cpu")
        """
        result = self.vino_model(input_data)[self.output_layer]
        return torch.from_numpy(result)

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.forward(input_data).to(self.device)
