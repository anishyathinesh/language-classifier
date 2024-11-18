
# does it contain popular nl fcn words? T - nl and F - end      
def nl_fcn_words(str):
    words = ['en', 'maar', 'als', 'op', 'aan']
    for w in words:
        if w in str.split():
            return True
    return False

# what are some substrings? T - nl and F - end      
def nl_frq_substrings(str):
    sub = ['ee', 'aa', 'oo', 'ij', 'vo']
    for w in sub:
        if w in str:
            return True
    return False
        
# is the avg word length > 5? T - nl and F - eng
def avg_word_length(str):
    str = str.split()
    sum = 0
    denom = 0
    for w in str:
        if len(w) > 3:
            denom += 1
            sum += len(w)

    avg = sum / denom

    if avg > 6:
        return True
    else:
        return False
    

# what are some substrings of length 2? T - nl and F - end      
def common_subs_2(str):
    phrases = ['de het', 'en het', 'in de', 'toe ten', 'was van']
    str = str.split()
    len2 = []
    for i in range(len(str)-1):
        len2.append(' '.join(str[i:i+2]))
    for p in phrases:
        if p in len2:
            return True
    return False
        
# what are some substrings of length 3? T - nl and F - end      
def common_subs_3(str):
    phrases = [
    "in de buurt",
    "het is een",
    "op de hoogte",
    "in het algemeen",
    "dat wegens zijn",
    "was van de"
    ]
    str = str.split()
    len3 = []
    for i in range(len(str)-2):
        len3.append(' '.join(str[i:i+3]))
    for p in phrases:
        if p in len3:
            return True 
    return False
