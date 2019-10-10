using FluxModels.RNNAutoencoder
using Flux

# dataset is An array of strings
strs = map(x->"$x", rand(1:10000, 1000))
# function to create a charlist. extract a unique charlist from strs
join_string = (x, y) -> begin
    unique([string.(unique(x))..., string.(unique(y))...])
end
charlist = [reduce(join_string, strs)..., "<SOS>", "<EOS>", "<PAD>"]

data = RNNAutoencoder.SequentialData(strs, charlist)

# latent vector dimension
features = 56
# model of RNNAutoencoder
s_ae = RNNAutoencoder.Autoencoder(features, data.n_char, data.n_seq)
# loss function of s_ae
loss = RNNAutoencoder.loss_func(s_ae)

batch_size = 50
for epoch in 1:10
    # Batch splitting randomly
    traindata = RNNAutoencoder.get_shuffled_traindata(data.sequence_data, batch_size)
    Flux.train!(loss, params(s_ae), traindata, ADAM())
    # Calculate loss with 100 sampled data
    @show loss(RNNAutoencoder.sample_data(data.sequence_data, 100)).data
end