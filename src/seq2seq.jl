

using Flux
using Random

"""
SequentialData
    n_data: number of data
    vstr: vector of strings
    charlist: list of char used in strings and added <SOS>, <EOS>, <PAD>
    n_seq: maximum number of sequence of string
    data: processed sequencial data
"""
mutable struct SequentialData
    n_data::Integer
    vstr::Vector{String}
    charlist::Vector{String}
    n_char::Integer
    n_seq::Integer
    data
    function SequentialData(vstr::Vector{String}, charlist::Vector{String})
        n_data = length(vstr)
        n_char = length(charlist)
        n_seq = maximum(map(length, vstr)) + 1
        vstr_token = map(str->push!(string.(collect(str)), "<EOS>"), vstr)
        data = map(str_token->append!(str_token, collect(Iterators.repeated("<PAD>", n_seq - length(str_token)))), vstr_token)
        data_array = reduce(hcat, data)
        data_sequential = [Flux.onehotbatch(data_array[i, :], charlist) for i in 1:n_seq]
        new(n_data, vstr, charlist, n_char, n_seq, data_sequential)
    end
end

"""
make dataset chunk, SequencialData splited by batch_size
"""
function get_shuffled_traindata(data::SequentialData, batch_size::Integer)
    n_data = length(data.vstr)
    vindices = collect(Iterators.partition(randperm(n_data), batch_size))
    return [([x[:, indices] for x in data.data],) for indices in vindices]
end

"""
from One-hot matrix to index of charlist
"""
onehot_to_indices(onehot) = reduce(vcat, mapslices(x->argmax(x), onehot, dims = 1))

struct Encoder
    embed
    gru
    function Encoder(features, n_char)
        embed = Flux.glorot_uniform(features, n_char) |> param
        gru = GRU(features, features)
        return new(embed, gru)
    end
end
Flux.@treelike Encoder # enable params(e::Encoder)

function (e::Encoder)(x)
    Flux.reset!(e.gru)
    vindices = map(onehot_to_indices, x)
    x = map(x1->e.embed[:, x1], vindices)
    hidden = e.gru.(x)[end]
    return hidden
end


struct Decoder
    embed
    gru
    dense
    n_char::Integer
    n_seq::Integer
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
    EOS = d.embed[:, [d.n_char - 1 for _ in 1:batch_size]]
    x = d.gru(EOS)
    x = d.dense(x)
    categorical = softmax(x)
    for i in 1:d.n_seq - 1
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
    n_char::Integer
    function Seq2Seq(features::Integer, data::SequentialData)
        encoder = Encoder(features, data.n_char)
        decoder = Decoder(features, data.n_char, data.n_seq)
        return new(encoder, decoder, data.n_char)
    end
end
Flux.@treelike Seq2Seq

function (m::Seq2Seq)(seq_x)
    hidden_state = m.encoder(reverse(seq_x))
    predict = m.decoder(hidden_state)
    return predict
end

function loss_func(m::Seq2Seq)
    loss_total = 0
    weight = ones(m.n_char)
    weight[end] = 0 # <PAD> weight to zero
    return function loss(seq_x)
        hidden_state = m.encoder(reverse(seq_x))
        predict = m.decoder(hidden_state)
        t = reduce(hcat, seq_x)
        loss_batch = Flux.crossentropy(predict, t, weight = weight)
        global loss_total += loss_batch.data
        return loss_batch
    end
end

