module RNNAutoencoder

using Flux
using Random
using Printf

struct SequentialData
    n_char
    n_seq
    charlist
    id_sos
    id_eos
    id_pad
    strs
    sequence_data

    function SequentialData(strs::Vector{String}, charlist::Vector{String})
        n_seq = maximum(map(length, strs)) + 2
        n_char = length(charlist)
        id_sos = n_char - 2
        id_eos = n_char - 1
        id_pad = n_char

        strs_edited = map(x -> ["<SOS>", string.(collect(x))..., "<EOS>"], strs)
        data = map(x->append!(x, collect(Iterators.repeated("<PAD>", n_seq-length(x)))), strs_edited)
        data_char = reduce(hcat, data)
        sequence_data = [Flux.onehotbatch(data_char[i, :], charlist) for i in 1:n_seq]

        return new(n_char, n_seq, charlist, id_sos, id_eos, id_pad, strs, sequence_data)
    end
end

"""
sample_data(sequence_data, N::Integer)
sampling N data from sequence_data

sampled_data = sample_data(sequence_data, 100)
loss = loss_func(m)
return data can use loss(sampled_data)
"""
function sample_data(sequence_data, N::Integer)
    n_data = size(sequence_data[1])[2]
    indices = randperm(n_data)[1:N]
    return [x[:, indices] for x in sequence_data]
end

function get_shuffled_traindata(sequence_data, batch_size)
    n_data = size(sequence_data[1])[2]
    vindices = collect(Iterators.partition(randperm(n_data), batch_size))
    return [([x[:, indices] for x in sequence_data],) for indices in vindices]
end

onehot_to_indices(onehot) = reduce(vcat, mapslices(x->argmax(x), onehot, dims=1))

struct Encoder
    embed
    gru
    n_char
    function Encoder(features, n_char)
        embed = Flux.glorot_uniform(features, n_char) |> param
        gru = GRU(features, features)
        return new(embed, gru, n_char)
    end
end
Flux.@treelike Encoder # params()の呼び出しを可能にする

function (e::Encoder)(x)
    Flux.reset!(e.gru)
    vindices = map(onehot_to_indices, x)
    x = map(x1 -> e.embed[:, x1], vindices)
    hidden = e.gru.(x)[end]
    return hidden
end

struct Decoder
    embed
    gru
    dense
    n_char
    n_seq
    function Decoder(features, n_char, n_seq)
        embed = Flux.glorot_uniform(features, n_char) |> param
        gru = GRU(features, features)
        dense = Dense(features, n_char, relu)
        return new(embed, gru, dense, n_char, n_seq)
    end
end
Flux.@treelike Decoder

function (d::Decoder)(h)
    Flux.reset!(d.gru)
    if length(size(h)) == 1
        batch_size = 1
    else
        _, batch_size = size(h)
    end
    d.gru.state = h
    sos = d.embed[:, [d.n_char-1 for _ in 1:batch_size]]
    x = d.gru(sos)
    x = d.dense(x)
    categorical = softmax(x)
    for i in 1:d.n_seq-2 # exclude <SOS>
        indices = onehot_to_indices(x)
        x2 = d.embed[:, indices]
        x = d.gru(x2) |> d.dense
        categorical = hcat(categorical, softmax(x))
    end
    categorical
end


struct Autoencoder
    encoder::Encoder
    decoder::Decoder
    function Autoencoder(features, n_char, n_seq)
        encoder = Encoder(features, n_char)
        decoder = Decoder(features, n_char, n_seq)
        return new(encoder, decoder)
    end
end
Flux.@treelike Autoencoder

function (m::Autoencoder)(seq_x)
    hidden_state = m.encoder(reverse(seq_x))
    predict = m.decoder(hidden_state)
    return predict
end

function loss_func(m::Autoencoder)
    weight = ones(m.decoder.n_char)
    weight[end] = 0 # end is <PAD> index
    return function loss(seq_x)
        hidden_state = m.encoder(reverse(seq_x))
        predict = m.decoder(hidden_state)
        t = reduce(hcat, map(x->x, seq_x[2:end])) # 1 is <SOS> so skip
        loss_batch = Flux.crossentropy(predict, t, weight=weight)
        return loss_batch
    end
end

function str2sequence(str, sdata::SequentialData)
    str = ["<SOS>", string.(collect(str))..., "<EOS>"]
    append!(str, collect(Iterators.repeated("<PAD>", sdata.n_seq - length(str))))
    onehot = Flux.onehotbatch(str, sdata.charlist)
    return [onehot[:, i] for i in 1:size(onehot)[2]]
end

function reconstruction_string(m::Autoencoder, sdata::SequentialData, str::AbstractString)
    seq_one = str2sequence(str, sdata)
    predict_onehot = m(seq_one)
    predict_strs = Flux.onecold(predict_onehot, sdata.charlist)
    eos_index = findfirst(x->x=="<EOS>", predict_strs)
    eos_index = eos_index === nothing ? sdata.n_seq : eos_index
    return reduce(*, predict_strs[1:eos_index-1])
end

"""
    z_to_string

    latent vector z to string
"""
function z_to_string(m::Autoencoder, sdata::SequentialData, z)
    predict_onehot = m.decoder(z)
    predict_strs = Flux.onecold(predict_onehot, sdata.charlist)
    eos_index = findfirst(x->x=="<EOS>", predict_strs)
    eos_index = eos_index === nothing ? sdata.n_seq : eos_index
    return reduce(*, predict_strs[1:eos_index-1])
end

end # module