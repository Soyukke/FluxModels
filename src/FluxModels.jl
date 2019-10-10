module FluxModels

using Random

include("rnn_autoencoder.jl")

function split_data(data::Vector, n_train)
    n_data = length(data)
    @assert n_data > n_train
    shuffled_data = shuffle(data)
    return shuffled_data[1:n_train], shuffled_data[n_train+1:end]
end

end # module
