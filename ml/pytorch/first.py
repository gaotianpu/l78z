import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def test_1(): 
    word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    print("embeds.weight", embeds.weight)
    lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
    hello_embed = embeds(lookup_tensor)
    print(hello_embed)
    print(dir(embeds))

def test_2(): 
    # We will use Shakespeare Sonnet 2
    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 10
    # n-gram language model we should tokenize the input, but we will ignore that for now
    # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
                for i in range(len(test_sentence) - 2)]
    # print the first 3, just so you can see what they look like
    print(trigrams[:3])
    vocab = set(test_sentence)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

def test_3():
    input=torch.randn(3,3)
    print("------",input)

    soft_input = torch.nn.Softmax(dim=0) 
    x = torch.log(soft_input(input)) 
    print("------",x)

    x1 = F.log_softmax(input, dim=0)
    print("------",x1)


if __name__ == "__main__":
    # test_1()
    # test_2()
    test_3()

