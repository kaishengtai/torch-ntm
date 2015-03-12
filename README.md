A Neural Turing Machine in Torch
================================

A Torch implementation of the Neural Turing Machine model described in this 
[paper](http://arxiv.org/abs/1410.5401) by Alex Graves, Greg Wayne and Ivo Danihelka.

This implementation uses an LSTM controller. NTM models with multiple read/write heads are supported.

## Requirements

[Torch7](https://github.com/torch/torch7) (of course), as well as the following
libraries:

[penlight](https://github.com/stevedonovan/Penlight)

[nn](https://github.com/torch/nn)

[optim](https://github.com/torch/optim)

[nngraph](https://github.com/torch/nngraph)

All the above dependencies can be installed using [luarocks](http://luarocks.org). For example:

```
luarocks install nngraph
```

## Usage

For the copy task:

```
th tasks/copy.lua
```

For the associative recall task:

```
th tasks/recall.lua
```
