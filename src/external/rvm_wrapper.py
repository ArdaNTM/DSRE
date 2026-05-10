import torch
import numpy as np

class RVMWrapper:
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device
        print(f"[RVM Wrapper] Model yükleniyor... ({self.device.upper()})")
        if model_path:
            self.model = torch.jit.load(model_path).to(self.device).eval()
        else:
            self.model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").to(self.device).eval()
            
        if 'cuda' in self.device:
            self.model = self.model.half()

        self.b0 = self.b1 = self.b2 = self.b3 = None

    def reset_hidden_states(self):
        self.b0 = self.b1 = self.b2 = self.b3 = None

    @torch.no_grad()
    def process_frame_tensor(self, frame_tensor: torch.Tensor, reset_state: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if reset_state:
            self.reset_hidden_states()

        fgr, pha, self.b0, self.b1, self.b2, self.b3 = self.model(
            frame_tensor, self.b0, self.b1, self.b2, self.b3
        )

        return fgr.squeeze(0), pha.squeeze(0)