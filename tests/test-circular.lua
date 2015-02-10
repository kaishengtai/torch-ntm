require('../')
require('torch')
require('nn')
require('sys')

function rotate_left(input, step)
  local output = input.new():resizeAs(input)
  local size = input:size(1)
  output[{{1, size - step}}] = input[{{step + 1, size}}]
  output[{{size - step + 1, size}}] = input[{{1, step}}]
  return output
end

local c = nn.CircularConvolution()
local a = torch.randn(15)
print(a)

local kernel = torch.Tensor{0, 0, 1}
print(c:forward{a, kernel})

kernel = torch.Tensor{1, 0, 0}
print(c:forward{a, kernel})

-- local a = torch.randn(10)
-- local b = torch.Tensor{-0.5, 0.5, 1, 2, 0.25}
-- local kernel = torch.zeros(10)
-- kernel[{{1, b:size(1)}}] = b
-- kernel = rotate_left(kernel, math.floor(b:size(1) / 2))
-- print(c:forward{a, b})
-- --print(c:updateOutput_orig{a, kernel})

-- local grad = torch.randn(10)
-- print(unpack(c:backward({a, b}, grad)))
--print(unpack(c:updateGradInput_orig({a, kernel}, grad)))

-- local start = sys.clock()
-- local dim = 100
-- for i = 1, 1000 do
--   local a = torch.randn(dim)
--   local b = torch.randn(dim)
--   c:forward{a, b}

--   local out_grad = torch.randn(dim)
--   c:backward({a, b}, out_grad)
-- end
-- print('done in ' .. (sys.clock() - start) .. 's')
