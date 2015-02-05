--[[

  Implementation of the Neural Turing Machine described here:

  http://arxiv.org/pdf/1410.5401v2.pdf

  Variable names take after the notation in the paper. Identifiers with "r"
  appended indicate read-head variables, and likewise for those with "w" appended.

--]]

local NTM, parent = torch.class('ntm.NTM', 'nn.Module')

function NTM:__init(input_dim, output_dim, mem_rows, mem_cols, cont_dim, shift_range)
  self.input_dim = input_dim
  self.output_dim = output_dim
  self.mem_rows = mem_rows
  self.mem_cols = mem_cols
  self.cont_dim = cont_dim
  self.shift_range = shift_range or 1

  self.depth = 0
  self.cells = {}
  self.master_cell = self:new_cell()
  self.init_module = self:new_init_module()

  self.gradInput = {
    torch.zeros(self.input_dim), -- input
    torch.zeros(self.mem_rows, self.mem_cols), -- M
    torch.zeros(self.mem_rows), -- wr
    torch.zeros(self.mem_rows), -- ww
    torch.zeros(self.mem_cols), -- r
    torch.zeros(self.cont_dim), -- m
    torch.zeros(self.cont_dim), -- c
  }

  local cell_params, _ = self.master_cell:parameters()
end

-- The initialization module initializes the state of NTM memory,
-- read/write weights, and the state of the LSTM controller.
function NTM:new_init_module()
  local dummy = nn.Identity()() -- always zero
  local output_init = nn.Tanh()(nn.Linear(1, self.output_dim)(dummy))
  local M_init_lin = nn.Linear(1, self.mem_rows * self.mem_cols)
  local M_init = nn.View(self.mem_rows, self.mem_cols)(
    nn.Tanh()(M_init_lin(dummy)))
  local wr_init_lin = nn.Linear(1, self.mem_rows)
  local wr_init = nn.SoftMax()(wr_init_lin(dummy))
  local ww_init_lin = nn.Linear(1, self.mem_rows)
  local ww_init = nn.SoftMax()(ww_init_lin(dummy))
  local r_init = nn.Tanh()(nn.Linear(1, self.mem_cols)(dummy))
  local m_init = nn.Tanh()(nn.Linear(1, self.cont_dim)(dummy))
  local c_init = nn.Tanh()(nn.Linear(1, self.cont_dim)(dummy))
  local inits = {output_init, M_init, wr_init, ww_init, r_init, m_init, c_init}
  local init_module = nn.gModule({dummy}, inits)

  -- We initialize the read and write distributions such that the
  -- weights decay exponentially over the rows of NTM memory.
  -- This sort of initialization seems to be important in my experiments (kst).
  wr_init_lin.bias:copy(torch.range(self.mem_rows, 1, -1))
  ww_init_lin.bias:copy(torch.range(self.mem_rows, 1, -1))

  return init_module
end

-- Create a new NTM cell. Each cell shares the parameters of the "master" cell
-- and stores the outputs of each iteration of forward propagation.
function NTM:new_cell()
  -- input to the network
  local input = nn.Identity()()

  -- vector read from memory
  local r_p = nn.Identity()()

  -- LSTM controller output
  local m_p = nn.Identity()()
  local c_p = nn.Identity()()

  -- previous memory state and read/write weights
  local M_p = nn.Identity()()
  local wr_p = nn.Identity()()
  local ww_p = nn.Identity()()

  -- output and hidden states of the controller module
  local m, c = self:new_controller_module(input, r_p, m_p, c_p)
  local M, wr, ww, r = self:new_mem_module(M_p, wr_p, ww_p, m)
  local output = self:new_output_module(m)

  local inputs = {input, M_p, wr_p, ww_p, r_p, m_p, c_p}
  local outputs = {output, M, wr, ww, r, m, c}

  local cell = nn.gModule(inputs, outputs)
  if self.master_cell ~= nil then
    share_params(cell, self.master_cell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return cell
end

-- Create a new LSTM controller
function NTM:new_controller_module(input, r_p, m_p, c_p)
  local new_gate = function()
    return nn.CAddTable(){
      nn.Linear(self.input_dim, self.cont_dim)(input),
      nn.Linear(self.mem_cols, self.cont_dim)(r_p),
      nn.Linear(self.cont_dim, self.cont_dim)(m_p)
    }
  end

  -- input, forget, and output gates
  local i = nn.Sigmoid()(new_gate())
  local f = nn.Sigmoid()(new_gate())
  local o = nn.Sigmoid()(new_gate())
  local update = nn.Tanh()(new_gate())

  -- update the state of the LSTM cell
  local c = nn.CAddTable(){
    nn.CMulTable(){f, c_p},
    nn.CMulTable(){i, update}
  }
  local m = nn.CMulTable(){o, nn.Tanh()(c)}
  return m, c
end

-- Create a new external memory cell
function NTM:new_mem_module(M_p, wr_p, ww_p, m)
  -- read head outputs
  local kr, sr, betar, gr, gammar = self:new_head(m)

  -- write head outputs
  local kw, sw, betaw, gw, gammaw = self:new_head(m)
  local a, e = self:new_add_erase_module(m)

  -- read address
  local wr, modules_r = self:new_addr_module(M_p, wr_p, kr, sr, betar, gr, gammar)

  -- write address
  local ww, modules_w = self:new_addr_module(M_p, ww_p, kw, sw, betaw, gw, gammaw)

  -- read vector from memory
  local r = nn.MixtureTable(){wr, M_p}

  -- erase some history from memory
  local Mtilde = nn.CMulTable(){
    M_p,
    nn.AddConstant(1)(nn.MulConstant(-1)(nn.OuterProd(){ww, e}))
  }

  -- write to memory
  local M = nn.CAddTable(){
    Mtilde,
    nn.OuterProd(){ww, a}
  }

  return M, wr, ww, r
end

-- Create a new head
function NTM:new_head(m)
  local k     = nn.Tanh()(nn.Linear(self.cont_dim, self.mem_cols)(m))
  local s     = nn.SoftMax()(nn.Linear(self.cont_dim, 2 * self.shift_range + 1)(m))
  local beta  = nn.SoftPlus()(nn.Linear(self.cont_dim, 1)(m))
  local g     = nn.Sigmoid()(nn.Linear(self.cont_dim, 1)(m))
  local gamma = nn.AddConstant(1)(
    nn.SoftPlus()(nn.Linear(self.cont_dim, 1)(m)))
  return k, s, beta, g, gamma
end

-- Create add/erase outputs for a write head
function NTM:new_add_erase_module(m)
  local a = nn.Tanh()(nn.Linear(self.cont_dim, self.mem_cols)(m))
  local e = nn.Sigmoid()(nn.Linear(self.cont_dim, self.mem_cols)(m))
  return a, e
end

-- Create a new addressing module for a head, given:
--  * the previous state of memory M_p
--  * the previous write weights w_p
--  * key vector k
--  * shift weights s = [ -c, -c+1, ... , 0, ..., c-1, c ]
--  * sharpening parameter beta
--  * gating parameter g
--  * attention focusing parameter gamma 
function NTM:new_addr_module(M_p, w_p, k, s, beta, g, gamma)
  -- 
  local sim = nn.SmoothCosineSimilarity(){M_p, k}
  local wc = nn.SoftMax()(nn.ScalarMulTable(){sim, beta})
  local wg = nn.CAddTable(){
    nn.ScalarMulTable(){wc, g},
    nn.ScalarMulTable(){w_p, nn.AddConstant(1)(nn.MulConstant(-1)(g))}
  }

  local wtilde = nn.CircularConvolution(){wg, s}
  local wpow = nn.PowTable(){wtilde, gamma}
  local w = nn.Normalize()(wpow)

  local modules = {sim, wc, wg, wtilde, wpow, k, s, beta, g, gamma}
  return w, modules
end

-- Create an output module, e.g. to output binary strings.
function NTM:new_output_module(m)
  local output = nn.Sigmoid()(nn.Linear(self.cont_dim, self.output_dim)(m))
  return output
end

-- Forward propagate one time step. The outputs of previous time steps are 
-- cached for backpropagation.
function NTM:forward(input)
  self.depth = self.depth + 1
  local cell = self.cells[self.depth]
  if cell == nil then
    cell = self:new_cell()
    self.cells[self.depth] = cell
  end
  
  local prev_outputs
  if self.depth == 1 then
    prev_outputs = self.init_module:forward(torch.Tensor{0})
  else
    prev_outputs = self.cells[self.depth - 1].output
  end

  -- get inputs
  local inputs = {input}
  for i = 2, #prev_outputs do
    inputs[i] = prev_outputs[i]
  end
  local outputs = cell:forward(inputs)
  self.output = outputs[1]
  return self.output
end

-- Backward propagate one time step. Throws an error if called more times than
-- forward has been called.
function NTM:backward(input, grad_output)
  if self.depth == 0 then
    error("No cells to backpropagate through")
  end
  local cell = self.cells[self.depth]
  local grad_outputs = {grad_output}
  for i = 2, #self.gradInput do
    grad_outputs[i] = self.gradInput[i]
  end

  -- get inputs
  local prev_outputs
  if self.depth == 1 then
    prev_outputs = self.init_module:forward(torch.Tensor{0})
  else
    prev_outputs = self.cells[self.depth - 1].output
  end
  local inputs = {input}
  for i = 2, #prev_outputs do
    inputs[i] = prev_outputs[i]
  end

  self.gradInput = cell:backward(inputs, grad_outputs)
  self.depth = self.depth - 1
  if self.depth == 0 then
    self.init_module:backward(torch.Tensor{0}, self.gradInput)
    for i = 1, #self.gradInput do
      self.gradInput[i]:zero()
    end
  end
  return self.gradInput
end

-- Get the state of memory
function NTM:get_memory(depth)
  if self.depth == 0 then
    return self.initial_values[2]
  end
  local depth = depth or self.depth
  return self.cells[self.depth].output[2]
end

-- Get read head weights over the rows of memory
function NTM:get_read_weights(depth)
  if self.depth == 0 then
    return self.initial_values[3]
  end
  local depth = depth or self.depth
  return self.cells[depth].output[3]
end

-- Get write head weights over the rows of memory
function NTM:get_write_weights(depth)
  if self.depth == 0 then
    return self.initial_values[4]
  end
  local depth = depth or self.depth
  return self.cells[depth].output[4]
end

-- Get the vector read from memory
function NTM:get_read_vector(depth)
  if self.depth == 0 then
    return self.initial_values[5]
  end
  local depth = depth or self.depth
  return self.cells[depth].output[5]
end

function NTM:parameters()
  local p, g = self.master_cell:parameters()
  local pi, gi = self.init_module:parameters()
  tablex.insertvalues(p, pi)
  tablex.insertvalues(g, gi)
  return p, g
end

function NTM:forget()
  self.depth = 0
  self:zeroGradParameters()
  for i = 1, #self.gradInput do
    self.gradInput[i]:zero()
  end
end

function NTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
  self.init_module:zeroGradParameters()
end
