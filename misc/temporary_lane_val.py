from alt_main import validate_lane
from multitask import YOLOL
import torch

state_dict = torch.load("./weights/best.pt", map_location=torch.device("cuda:1"))

print(state_dict["net"])
