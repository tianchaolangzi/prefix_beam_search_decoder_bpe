import numpy as np
from ctc_decoders import Scorer, ctc_beam_search_decoder


def softmax(x):
  m = np.expand_dims(np.max(x, axis=-1), -1)
  e = np.exp(x - m)
  return e / np.expand_dims(e.sum(axis=-1), -1)


def main():
    logp_file = "./log_prob_1.pt.npy"
    lm_path = "/data/zhoukai/asr-model/english-1.1/english_cloud_lm_0310.binary"
    vocabulary = [
        "<unk>", "##e", "##s", "▁", "##t", "##a", "##o", "##i", "the", "##d", 
        "##l", "##n", "a", "##m", "##y", "##u", "s", "##p", "##ed", "##c", 
        "and", "##re", "to", "of", "##r", "##w", "##ing", "w", "##h", "p", 
        "c", "##er", "##f", "##k", "##ar", "in", "f", "b", "##g", "##an", 
        "##in", "i", "##en", "he", "##le", "g", "##or", "##ll", "##b", "be", 
        "##ro", "##st", "##on", "d", "##v", "##ly", "##ce", "##ur", "##es", "that", 
        "o", "##us", "was", "it", "th", "##ve", "##ch", "##un", "##al", "t", 
        "ma", "##ri", "you", "on", "##ver", "##ent", "for", "re", "##ra", "##'", 
        "his", "##ir", "##ter", "with", "her", "##it", "##th", "mo", "me", "ha", 
        "e", "as", "##tion", "had", "not", "no", "do", "##ther", "but", "st", 
        "she", "is", "##igh", "ho", "lo", "##ng", "him", "an", "##ck", "##j", 
        "##ugh", "de", "li", "mi", "la", "my", "con", "have", "this", "which", 
        "##q", "up", "said", "from", "who", "ex", "##x", "##z"
    ]
    vocabulary_p = []
    start_tokens = []
    for index, token in enumerate(vocabulary):
        if token.startswith("##"):
            vocabulary_p.append(token[2:])
        elif token == "<unk>":
            vocabulary_p.append(token)
        elif token == "▁":
            vocabulary_p.append(token)
            start_tokens.append(index)
        else:
            vocabulary_p.append("▁"+token)
            start_tokens.append(index)
    
    # vocab_char = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
    vocab_size = len(vocabulary_p) + 1
    logp = np.load(logp_file)
    assert vocab_size == logp.shape[1]

    scorer = Scorer(alpha=2.0, beta=0.5, model_path=lm_path, vocabulary=vocabulary_p)
    # print(scorer.get_dict_size())
    res = ctc_beam_search_decoder(softmax(logp), vocabulary_p, start_tokens, beam_size=32, ext_scoring_func=scorer)
    res_prob, decoded_text = res[0]

    print(decoded_text)
    # "could you help me calculate the result of twelve times twelve"



if __name__ == "__main__":
    main()
