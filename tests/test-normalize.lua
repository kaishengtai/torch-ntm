require('../')
require('torch')
require('nn')
require('sys')

local n = nn.Normalize()
local x = torch.Tensor{1, 2, 3, 4}
print(n:forward(x))
print(n:backward(x, x))
