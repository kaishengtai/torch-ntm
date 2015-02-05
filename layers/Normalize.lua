--[[
  
 Divides each element of a Tensor by their sum.

--]]

local Normalize, parent = torch.class('nn.Normalize', 'nn.Module')

function Normalize:__init()
  parent.__init(self)
end

function Normalize:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  self.sum = input:sum()
  self.output:div(self.sum)
  return self.output
end

function Normalize:updateGradInput(input, gradOutput)
  local size = input:size(1)
  self.gradInput:resizeAs(input)
  for i = 1, size do
    local output = torch.Tensor(size):copy(self.output)
    output:div(-self.sum)
    output[i] = output[i] + (1 / self.sum)
    self.gradInput[i] = gradOutput:dot(output)
  end
  return self.gradInput
end