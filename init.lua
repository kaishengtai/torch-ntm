require('torch')
require('nn')
require('nngraph')

ntm = {}
printf = utils.printf

include('rmsprop.lua')
include('layers/CircularConvolution.lua')
include('layers/Normalize.lua')
include('layers/OuterProd.lua')
include('layers/PowTable.lua')
include('layers/Print.lua')
include('layers/SmoothCosineSimilarity.lua')
include('layers/ScalarMulTable.lua')
include('layers/ScalarDivTable.lua')
include('NTM.lua')

function share_params(cell, src, ...)
  for i = 1, #cell.forwardnodes do
    local node = cell.forwardnodes[i]
    if node.data.module then
      node.data.module:share(src.forwardnodes[i].data.module, ...)
    end
  end
end
