require('../')
require('torch')
require('nn')

local o = nn.OuterProd()
local u = torch.randn(4)
local v = torch.randn(3)
local w = torch.randn(2)

print(o:forward{u, v})
print(unpack(o:backward({u, v}, torch.randn(4, 3))))

print(o:forward{v, w})
print(unpack(o:backward({v, w}, torch.randn(3, 2))))

print(o:forward{u, w})
print(unpack(o:backward({u, w}, torch.randn(4, 2))))

print(o:forward{u, v, w})
print(unpack(o:backward({u, v, w}, torch.randn(4, 3, 2))))
