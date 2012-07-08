
-----------------------------------------------------------------------
print '==> constructing model'

if params.model == 'linear' then

   -- params
   inputSize = params.inputsize*params.inputsize
   outputSize = params.nfiltersout

   -- encoder
   encoder = nn.Sequential()
   encoder:add(nn.Linear(inputSize,outputSize))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputSize))

   -- decoder
   decoder = nn.Sequential()
   decoder:add(nn.Linear(outputSize,inputSize))

   -- complete model
   module = unsup.AutoEncoder(encoder, decoder, params.beta)

   -- verbose
   print('==> constructed linear auto-encoder')

elseif params.model == 'conv' then

   -- params:
   conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   kw, kh = params.kernelsize, params.kernelsize
   iw, ih = params.inputsize, params.inputsize

   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }]
   local outputFeatures = conntable[{ {},2 }]:max()

   -- encoder:
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputFeatures))

   -- decoder:
   decoder = nn.Sequential()
   decoder:add(nn.SpatialFullConvolutionMap(decodertable, kw, kh, 1, 1))

   -- complete model
   module = unsup.AutoEncoder(encoder, decoder, params.beta)

   -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
   dataset:conv()

   -- verbose
   print('==> constructed convolutional auto-encoder')

elseif params.model == 'linear-psd' then

   -- params
   inputSize = params.inputsize*params.inputsize
   outputSize = params.nfiltersout

   -- encoder
   encoder = nn.Sequential()
   encoder:add(nn.Linear(inputSize,outputSize))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputSize))

   -- decoder is L1 solution
   decoder = unsup.LinearFistaL1(inputSize, outputSize, params.lambda)

   -- PSD autoencoder
   module = unsup.PSD(encoder, decoder, params.beta)

   -- verbose
   print('==> constructed linear predictive sparse decomposition (PSD) auto-encoder')

elseif params.model == 'conv-psd' then

   -- params:
   conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   kw, kh = params.kernelsize, params.kernelsize
   iw, ih = params.inputsize, params.inputsize

   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }]
   local outputFeatures = conntable[{ {},2 }]:max()

   -- encoder:
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputFeatures))

   -- decoder is L1 solution:
   decoder = unsup.SpatialConvFistaL1(decodertable, kw, kh, iw, ih, params.lambda)

   -- PSD autoencoder
   module = unsup.PSD(encoder, decoder, params.beta)

   -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
   dataset:conv()

   -- verbose
   print('==> constructed convolutional predictive sparse decomposition (PSD) auto-encoder')

else
   print('==> unknown model: ' .. params.model)
   os.exit()
end
