using Random


const TRAINPRC = 0.7


mutable struct Dataset
    path
    categories
    categoryprob
    train
    test
    Dataset(path::String, categories::Tuple) = (x = new(path, categories); x.categoryprob = 1 / length(x.categories);(x.train, x.test) = datasetsplit(datafolderpath, categories); return x) 
end


function datasetsplit(datafolderpath, categories)
    trainprc = TRAINPRC
    
    filesincat = Dict(cat => readdir(datafolderpath * cat ) for cat in categories) 
    train = Dict{String, Vector{String}}()
    test = Dict{String, Vector{String}}()
    
    # divide into train and test sets
    for cat in categories
        
        cattrainlen::Integer = round(trainprc * length(filesincat[cat]))
        train[cat] = filesincat[cat][1:cattrainlen]    
        test[cat] = filesincat[cat][cattrainlen+1:end]
        
    end
    
    return train, test
end

function vectorize(filepath, lenofhash=2^10)

    # create vocabulary vectory initialized with zeros
    vocab = [0 for i=1:lenofhash]

    # read file, split it on spaces, change every letter to lowercase
    lwords = read(filepath, String) |> split .|> lowercase

    # for every word create a hash of given length than increment value at that index in vocabulary vector
    for i in lwords
        vocab[(hash(i) % lenofhash) + 1] += 1
    end

    return vocab
end

function train(data::Dataset, lenofhash=2^10, α=0.01)
    # create vocabulary dictionary specifing probability of each word appearing in each of the categories

    vocabdict = Dict{String, Vector{Float64}}()
    
    for cat in data.categories
    
        avrvocab = [0.0 for i=1:lenofhash]
    
        # sum word appearances in files of given category
        for file in data.train[cat]
            filepath = data.path * cat * "/" * file
            newvocab = vectorize(filepath, lenofhash)
            avrvocab += newvocab
        end

        suma = sum(avrvocab)
        # use relative frequency counting to establish maximum likelihood of every words
        # α is a smoothing factor used to eliminate zero probabilities
        for (i, _) in enumerate(avrvocab)
            avrvocab[i] = (avrvocab[i] + α) / (suma + lenofhash * α)
        end


        vocabdict[cat] = avrvocab
    end

    return vocabdict
end

function inference(data::Dataset, vocabdict, filepath, lenofhash)
    
    samplevectorized = vectorize(filepath, lenofhash)
    onehotevocab = [samplevectorized[i] > 0 ? 1.0 : 0.0 for i=1:lenofhash]

    categoryprob = []
    for cat in data.categories
        probabilities = onehotevocab .* vocabdict[cat]
        prob = data.categoryprob
        for i in probabilities
            if i > 0
                #multiply probabilities of words appearing in the sample and the probability of given category
                prob = prob * i * 100
            end
        end
        append!(categoryprob, prob)
    end

    res = categories[argmax(categoryprob)]
end

function test(data::Dataset,vocabdict, lenofhash=2^10)

    testresult = Dict{String, Number}()

    for cat in data.categories
        testlen = length(data.test[cat])
        positive = 0
        for file in data.test[cat]
            filepath = data.path * cat * "/" * file
            res = inference(data,vocabdict, filepath, lenofhash)
            if res == cat
                positive +=1
            end
        end
        testresult[cat] = positive / testlen
        println("Test result in category $cat : $(testresult[cat]) ")
    end
    
end

function randomtest(data::Dataset, vocabdict, lenofhash=2^10)

    cat = rand(data.categories)
    filepath = data.path * cat * "/" * rand(data.test[cat])
    println(read(filepath, String))
    res = inference(data, vocabdict, filepath, lenofhash)
    println("Network category: $res Real category: $cat")
    
end

function customtest(data::Dataset,vocabdict, filepath, lenofhash=2^10)

    println(read(filepath, String))
    res = inference(data, vocabdict,filepath, lenofhash)
    println("Network category: $res")

end



datafolderpath = "/home/dominik/julia/bayesiannet/bbc/"
categories = ("business", "entertainment", "politics", "sport",  "tech")
dataset = Dataset(datafolderpath, categories)
println(length(dataset.test["sport"]))

vocabdict = train(dataset)
test(dataset, vocabdict)
# randomtest(dataset,vocabdict)
customtest(dataset,vocabdict, "/home/dominik/julia/bayesiannet/test.txt")
