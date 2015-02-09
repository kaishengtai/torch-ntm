function argmax(x)
  local index = 1
  local max = x[1]
  for i = 2, x:size(1) do
    if x[i] > max then
      index = i
      max = x[i]
    end
  end
  return index, max
end

function print_read_max(model)
  local read_weights = model:get_read_weights()
  local num_heads = model.read_heads
  local fmt = '%-4d %.4f'
  if num_heads == 1 then
    printf(fmt .. '\n', argmax(read_weights))
  else
    local s = ''
    for i = 1, num_heads do
      s = s .. string.format(fmt, argmax(read_weights[i]))
      if i < num_heads then s = s .. ' | ' end
    end
    print(s)
  end
end

function print_write_max(model)
  local write_weights = model:get_write_weights()
  local num_heads = model.write_heads
  local fmt = '%-4d %.4f'
  if num_heads == 1 then
    printf(fmt .. '\n', argmax(write_weights))
  else
    local s = ''
    for i = 1, num_heads do
      s = s .. string.format(fmt, argmax(write_weights[i]))
      if i < num_heads then s = s .. ' | ' end
    end
    print(s)
  end
end
