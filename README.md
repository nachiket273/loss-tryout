# loss-tryout
PyTorch implementation of <a href=https://arxiv.org/pdf/2204.12511.pdf>POLYLOSS: A POLYNOMIAL EXPANSION PERSPECTIVE OF CLASSIFICATION LOSS FUNCTIONS</a>
<br>
<br>
polyloss.py contains the implementation of Poly-1 loss and Poly-1 focal loss. <br>
While polynloss.py contains simple implementations of Poly-N and Poly-N focal loss. Note that this implemenation of Poly-N loss is just for experimenation. <br>
<br>
Simple_Example_Polyloss.ipynb is notebook with simple examples of Poly-1 loss usage.

## Usage

1) Git clone the repoository and change to directory
```Python
git clone https://github.com/nachiket273/loss-tryout.git
cd loss-tryout
```

2) Import
```Python
from polyloss import PolyLoss, PolyFocalLoss
```

3) Poly crossentropy loss example
```Python
import torch
inputs = torch.rand((20, 5))
targets = torch.randint(high=5, size=(20,))
ploss = PolyLoss()
ploss(inputs, targets)
```

4) Poly focal loss example
```
import torch
inputs = torch.rand((20, 5))
targets = torch.randint(high=5, size=(20,))
pfloss = PolyFocalLoss()
pfloss(inputs, targets)
```

## Citations

```bibtex
@misc{POLYLOSS,
    title={POLYLOSS: A POLYNOMIAL EXPANSION PERSPECTIVE OF CLASSIFICATION LOSS FUNCTIONS},
    author={Zhaoqi Leng, Mingxing Tan, Chenxi Liu, Ekin Dogus Cubuk, Xiaojie Shi, Shuyang Cheng, Dragomir Anguelov},
    year={2022},
    url={https://arxiv.org/pdf/2204.12511.pdf},
}
```
