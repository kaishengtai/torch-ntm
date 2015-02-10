require('../')
require('torch')
require('nn')

local c = nn.SmoothCosineSimilarity()
local M = torch.randn(10, 4)
--M[1]:fill(0)
local k = torch.randn(4)
local output = c:forward{M, k}
local out_grad = torch.randn(10)
local Mgrad, kgrad = unpack(c:backward({M, k}, out_grad))
print(output)
print(Mgrad)
print(kgrad)

print('comparison with nn.CosineDistance')
local d = nn.CosineDistance()
local sum_dkgrad = torch.Tensor(4):zero()
for i = 1, M:size(1) do
  print('row ' .. i)
  local doutput = d:forward{M[i], k}
  local dMgrad, dkgrad = unpack(d:backward({M[i], k}, out_grad:narrow(1, i, 1)))
  print((output:narrow(1, i, 1) - doutput):abs():max())
  print((Mgrad[i] - dMgrad):abs():max())
  sum_dkgrad:add(dkgrad)
end
print('k grad difference')
print((kgrad - sum_dkgrad):abs():max())
