"""
An example of left->right propagation

Other direction settings:
left->right: Propagator = GateRecurrent2dnoind(True,False)
right->left: Propagator = GateRecurrent2dnoind(True,True)
top->bottom: Propagator = GateRecurrent2dnoind(False,False)
bottom->top: Propagator = GateRecurrent2dnoind(False,True)

X: any signal/feature map to be filtered
G1~G3: three coefficient maps (e.g., left-top, left-center, left-bottom)

Note:
1. G1~G3 constitute the affinity, they can be a bounch of output maps coming from any CNN, with the input of any useful known information (e.g., RGB images)
2. for any pixel i, |G1(i)| + |G2(i)| + |G3(i)| <= 1 is a sufficent condition for model stability (see paper)
"""
import torch
from torch.autograd import Variable
from pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind

Propagator = GateRecurrent2dnoind(True,False)

X = Variable(torch.randn(1,3,10,10))
G1 = Variable(torch.randn(1,3,10,10))
G2 = Variable(torch.randn(1,3,10,10))
G3 = Variable(torch.randn(1,3,10,10))

sum_abs = G1.abs() + G2.abs() + G3.abs()
mask_need_norm = sum_abs.ge(1)
mask_need_norm = mask_need_norm.float()
G1_norm = torch.div(G1, sum_abs)
G2_norm = torch.div(G2, sum_abs)
G3_norm = torch.div(G3, sum_abs)

G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm

X = X.cuda()
G1 = G1.cuda()
G2 = G2.cuda()
G3 = G3.cuda()

output = Propagator.forward(X,G1,G2,G3)
print(X)
print(output)
