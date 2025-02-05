import torch
import torch.nn.functional as F
from pprint import pprint 

# Python 리스트를 PyTorch 텐서로 변환
state = torch.tensor([1.2, 0.5, -0.7, 0.1], dtype=torch.float) 

pprint(F.softmax(state, dim = 0))
print('\n')

states = torch.tensor([
    [1,2, 0.3,-0.4, 2],
    [1,2, 0.3,-0.4, 2],
    [1,2, 0.3,-0.4, 2]
    ],dtype=torch.float)

pprint(F.softmax(states, dim = 1))
print('\n')
# pprint(F.softmax(state, dim = 2))
