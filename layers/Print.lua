--[[

 An Identity layer that prints its input.

--]]

local Print, parent = torch.class('nn.Print', 'nn.Module')

function Print:__init(label)
  parent:__init(self)
  self.label = label
end

function Print:updateOutput(input)
  self.output = input
  if self.label ~= nil then
    print(self.label)
  end
  print(input)
  return self.output
end


function Print:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end
