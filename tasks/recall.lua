--[[

  Training a NTM for associative recall.

  More precisely:
  (i)   the NTM is presented a sequence (x_1, x_2, ..., x_k), where each item x_i
        is a sequence of length-m bitstrings.
  (ii)  the NTM is queried with an item x_i, 1 <= i <= k-1.
  (iii) the NTM is expected to output x_{i+1}, the following item in the
        training sequence.

  This task is a bit different from the one described in the paper since we
  can make more than one query after each training sequence.

  The model should converge to the optimum after about 20000 iterations.

--]]

require('../')
require('./util')
require('sys')

torch.manualSeed(0)

local config = {
  input_dim = 8,
  output_dim = 8,
  mem_rows = 128,
  mem_cols = 20,
  cont_dim = 100
}

local input_dim = config.input_dim

-- delimiter symbol and query symbol
local delim_symbol = torch.zeros(input_dim)
delim_symbol[1] = 1
local query_symbol = torch.zeros(input_dim)
query_symbol[2] = 1

function generate_items(num_items, item_len)
  local items = {}
  for i = 1, num_items do
    local item = torch.rand(item_len, input_dim):round()
    for j = 1, item_len do
      item[{j, {1, 2}}]:zero()
    end
    table.insert(items, item)
  end
  return items
end

function forward(model, items, num_queries, print_flag)
  local num_items = #items
  local item_len = items[1]:size(1)
  local loss = 0

  -- present items
  if print_flag then
    print(string.rep('-', 40))
    print('presentation phase: writes')
  end
  for i = 1, num_items do
    model:forward(delim_symbol)
    for j = 1, item_len do
      model:forward(items[i][j])
      if print_flag then
        printf('w: %d\t%.4f\n', argmax(model:get_write_weights()))
      end
    end
  end

  -- present queries
  local zeros = torch.zeros(input_dim)
  local outputs = {}
  local criteria = {}
  local query_indices = {}
  if print_flag then
    print(string.rep('-', 40))
    print('query phase: reads/writes')
  end
  for i = 1, num_queries do
    criteria[i] = {}
    outputs[i] = torch.Tensor(item_len, input_dim)

    local query_idx = math.floor(torch.uniform(1, num_items))
    query_indices[i] = query_idx
    local query = items[query_idx]
    local target = items[query_idx + 1]
    
    -- query
    model:forward(query_symbol)
    if print_flag then
      printf('-- query %d of %d: index %d\n', i, num_queries, query_idx)
      print(query)
    end
    for j = 1, item_len do
      model:forward(query[j])
      if print_flag then
        local ri, rw = argmax(model:get_read_weights())
        local wi, ww = argmax(model:get_write_weights())
        printf('r: %d\t%.4f\t|\tw: %d\t%.4f\n', ri, rw, wi, ww)
      end
    end

    -- target
    model:forward(query_symbol)
    if print_flag then
      printf('-- target %d of %d: index %d\n', i, num_queries, query_idx + 1)
      print(target)
    end
    for j = 1, item_len do
      criteria[i][j] = nn.BCECriterion()
      outputs[i][j] = model:forward(zeros)
      loss = loss + criteria[i][j]:forward(outputs[i][j], target[j]) * input_dim
      if print_flag then
        printf('r: %d\t%.4f\n', argmax(model:get_read_weights()))
      end
    end
    if print_flag then
      printf('-- output %d of %d\n', i, num_queries)
      print(outputs[i])
    end
  end
  return query_indices, outputs, criteria, loss
end

function backward(model, items, query_indices, outputs, criteria)
  local num_queries = #query_indices
  local num_items = #items
  local item_len = items[1]:size(1)
  local zeros = torch.zeros(input_dim)
  for i = num_queries, 1, -1 do
    local query_idx = query_indices[i]
    local query = items[query_idx]
    local target = items[query_idx + 1]

    -- target
    for j = item_len, 1, -1 do
      model:backward(
        zeros, criteria[i][j]:backward(outputs[i][j], target[j]):mul(input_dim))
    end
    model:backward(query_symbol, zeros)

    -- query
    for j = item_len, 1, -1 do
      model:backward(query[j], zeros)
    end
    model:backward(query_symbol, zeros)
  end

  for i = num_items, 1, -1 do
    local item = items[i]
    for j = item_len, 1, -1 do
      model:backward(item[j], zeros)
    end
    model:backward(delim_symbol, zeros)
  end
end

local model = ntm.NTM(config)
local params, grads = model:getParameters()
local num_iters = 50000
local min_len = 2
local max_len = 6
local item_len = 3
print(string.rep('=', 80))
print("NTM associative recall task")
print('training up to ' .. num_iters .. ' iteration(s)')
print('min sequence length = ' .. min_len)
print('max sequence length = ' .. max_len)
print('sequence element length = ' .. item_len)
print(string.rep('=', 80))
print('num params: ' .. params:size(1))

local rmsprop_state = {
  learningRate = 1e-4,
  momentum = 0.9,
  decay = 0.95
}

-- train
local start = sys.clock()
local print_interval = 25
for iter = 1, num_iters do
  local print_flag = (iter % print_interval == 0)
  local num_items = math.floor(torch.uniform(min_len, max_len + 1))
  local num_queries = math.floor(torch.uniform(1, num_items + 1))
  local items = generate_items(num_items, item_len)

  local feval = function(x)
    if print_flag then
      print(string.rep('=', 80))
      print('iter = ' .. iter)
      print('learn rate = ' .. rmsprop_state.learningRate)
      print('momentum = ' .. rmsprop_state.momentum)
      print('decay = ' .. rmsprop_state.decay)
      print('num items = ' .. num_items)
      print('num queries = ' .. num_queries)
      printf('t = %.1fs\n', sys.clock() - start)
    end

    local loss = 0
    grads:zero()

    local query_indices, outputs, criteria, sample_loss = forward(
      model, items, num_queries, print_flag)
    loss = loss + sample_loss
    backward(model, items, query_indices, outputs, criteria)

    -- clip gradients
    grads:clamp(-10, 10)
    if print_flag then
      print('max grad = ' .. grads:max())
      print('min grad = ' .. grads:min())
      print('loss = ' .. loss)
    end
    return loss, grads
  end

  ntm.rmsprop(feval, params, rmsprop_state)
end
