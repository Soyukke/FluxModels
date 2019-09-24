module RNNAutoencoder

import Flux:onehotbatch
using Flux
using Flux:reset!
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
        sequence_data = [onehotbatch(data_char[i, :], charlist) for i in 1:n_seq]

        return new(n_char, n_seq, charlist, id_sos, id_eos, id_pad, strs, sequence_data)
    end
end

function get_shuffled_traindata(sequence_data, batch_size)
    n_data = size(data.sequence_data[1])[2]
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
    reset!(e.gru)
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
    reset!(d.gru)
    _, batch_size = size(h)
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


struct Seq2Seq
    encoder::Encoder
    decoder::Decoder
    function Seq2Seq(features, n_char, n_seq)
        encoder = Encoder(features, n_char)
        decoder = Decoder(features, n_char, n_seq)
        return new(encoder, decoder)
    end
end
Flux.@treelike Seq2Seq

loss_total = 0
function loss_func(m::Seq2Seq)
    weight = ones(m.decoder.n_char)
    weight[end] = 0
    return function loss(seq_x)
        hidden_state = m.encoder(reverse(seq_x))
        predict = m.decoder(hidden_state)
        t = reduce(hcat, map(x->x, seq_x[2:end]))
        loss_batch = Flux.crossentropy(predict, t, weight=weight)
        global loss_total += loss_batch
        return loss_batch
    end
end

end # module