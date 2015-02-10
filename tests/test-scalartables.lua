require('../')

local v = torch.Tensor{1,2,3}
local c = torch.Tensor{1.5}

print('ScalarMulTable')
local mul = nn.ScalarMulTable()
print(mul:forward{v, c})
print(unpack(mul:backward({v, c}, torch.Tensor{1, 2, 2})))

print('ScalarDivTable')
local div = nn.ScalarDivTable()
print(div:forward{v, c})
print(unpack(div:backward({v, c}, torch.Tensor{1, 2, 2})))

print('PowTable')
local pow = nn.PowTable()
print(pow:forward{v, c})
print(unpack(pow:backward({v, c}, torch.Tensor{1, 2, 2})))
