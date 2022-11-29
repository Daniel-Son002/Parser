import nltk
import sys
import os
import numpy as np
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    contents = dict()

    for root, dirnames, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                content = open(file_path, encoding="utf8")
                contents[file] = content.read()
    return contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    strings = nltk.word_tokenize(document.lower())
    lst = []

    for word in strings:
        if word not in nltk.corpus.stopwords.words("english") and \
                not all(temp in string.punctuation for temp in word):
            lst.append(word)

    return lst


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    flat_lst = []
    for lst in documents.values():
        for word in set(lst):
            flat_lst.append(word)

    for word in flat_lst:
        word_count = flat_lst.count(word)
        idf = np.log(len(documents.keys()) / word_count)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    values = []
    for key, lst in files.items():
        tot = 0
        for word in query:
            tot += lst.count(word) * idfs[word]
        values.append((key, tot))
    new_list = sorted(values, key=lambda x: x[1], reverse=True)
    lst = [element[0] for element in new_list[:n]]
    return [element[0] for element in new_list[:n]]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    values = []
    for sentence in sentences:
        idf_tot = 0
        tie = 0
        for word in query:
            if word in sentences[sentence]:
                idf_tot += idfs[word]
                tie += \
                    sentences[sentence].count(word) / len(sentences[sentence])
        values.append((sentence, idf_tot, tie))

    ranks = sorted(values, key=lambda x: (x[1], x[2]), reverse=True)
    return [element[0] for element in ranks[:n]]


if __name__ == "__main__":
    main()
