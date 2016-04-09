text_language = "french"
input_file_name = "input.txt"
output_file_name = "output.txt"


def read_from_file(file_name):
    return open(file_name,'r').read().splitlines()


def create_pairs(text):
    # a b c d -> (a b), (b c), (c d)
    return list(zip(text[:-1], text[1:]))


def get_tokens(paragraph):
    import nltk
    import string
    from nltk.corpus import stopwords
    lower_case = paragraph.lower()
    no_punctuation = lower_case.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    no_digits = no_punctuation.translate(str.maketrans(string.digits, " " * len(string.digits)))
    tokens = nltk.word_tokenize(no_digits)
    return [x for x in tokens if x not in stopwords.words(text_language)]


def get_tfidf_vector_pair(pair):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(tokenizer=get_tokens)
    return tfidf.fit_transform([pair[0], pair[1]])


def get_cosines(text):
    from sklearn.metrics.pairwise import cosine_similarity
    cosines = []
    pairs = create_pairs(text)
    for pair in pairs:
        tfidf_pair = get_tfidf_vector_pair(pair)
        cosines.append(cosine_similarity(tfidf_pair[0], tfidf_pair[1])[0, 0])
    return cosines


def get_depth_score(array):
    res = [2*(array[1]-array[0])]
    for i in range(1, len(array)-1):
        res.append(array[i-1] + array[i+1] - 2 * array[i])
    return res


def smooth(array):
    length = len(array)
    cycles = 4
    radius = 1
    diameter = radius * 2 + 1
    result = array.copy()
    for k in range(cycles):
        for i in range(radius, length - radius):
            sum = result[i]
            for j in range(1, radius + 1):
                sum += result[i-j]
                sum += result[i+j]
            result[i] = sum / diameter
        for i in range(0, radius):
            sum = result[i]
            num = 1
            for j in range(0,i):
                sum += result[j]
                sum += result[2*i-j]
                num += 2
            result[i] = sum / num
        for i in range(length-radius, length):
            sum = result[i]
            num = 1
            for j in range(i, length):
                sum += result[j]
                sum += result[2 * i - j]
                num += 2
            result[i] = sum / num
    return result


def get_local_minimums(array):
    result = [0]*len(array)
    if array[0] < array[1]:
        result[0] = 1
    for i in range(1, len(array) - 2):
        if array[i] < min(array[i+1],array[i-1]):
            result[i] = 1
    if array[len(array)-1] < array[len(array)-2]:
        result[len(result)-1] = 1
    return result


def re_build_text(text, minimums):
    result = []
    paragraph = text[0]
    for i in range(0, len(minimums)):
        if minimums[i] == 1:
            result.append(paragraph)
            paragraph = text[i+1]
        else:
            paragraph += " " + text[i+1]
    if paragraph != "":
        result.append(paragraph)
    return result


import matplotlib.pyplot as plt
text = read_from_file(input_file_name)
y = get_cosines(text)
s = smooth(y)
d = get_depth_score(s)
plt.subplot(211)
plt.plot(range(0,len(y)),y, color='blue')
plt.plot(range(0,len(s)),s, color='black')
plt.subplot(212)
plt.plot(range(0,len(d)),d, color='red')
minimums = get_local_minimums(d)
new_text = re_build_text(text,minimums)

print(len(new_text))
print(minimums)
plt.show()

output = open(output_file_name, "w")
for i in new_text:
    output.write(i + "\n\n")
output.close()
