from collections import Counter
import math

def bleu_score(candidate, reference, max_n):

    def generate_ngrams(text, n):
        return [tuple(text[i:i+n]) for i in range(len(text)-n+1)]

    def modified_precision(candidate, reference, n):
        c_ng = Counter(generate_ngrams(candidate, n))
        r_ng = Counter(generate_ngrams(reference, n))
        numerator   = sum(min(c_ng[ng], r_ng[ng]) for ng in c_ng)
        denominator = sum(c_ng.values())              
        return numerator / denominator if denominator > 0 else 0

    def brevity_penalty(candidate, reference):        
        c = len(candidate)                   
        r = len(reference)                    
        if c >= r:
            return 1
        else:
            return math.exp(1 - (r/c))               

    precisions = []
    for i in range(1, max_n+1):
        p = modified_precision(candidate, reference, i)
        if p == 0:
            return 0                                   
        precisions.append(math.log(p))

    bp = brevity_penalty(candidate, reference)
    return bp * math.exp((1/max_n) * sum(precisions))