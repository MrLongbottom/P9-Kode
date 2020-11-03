

def load_vector_file(path, seperator=','):
    with open(path, 'r') as file:
        dictionary = {}
        for line in file.readlines():
            kv = line.split(seperator)
            value = kv[1].replace('\n', '')
            if len(value.split(';')) > 1:
                value = value.split(';')
            dictionary[int(kv[0])] = value
    return dictionary


def save_vector_file(filename, content, seperator=','):
    """
    Saves content of list as a vector in a file, similar to a Word2Vec document.
    :param filename: path of file to save.
    :param content: list of content to save.
    :return: None
    """
    print('Saving file "' + filename + '".')
    with open(filename, "w") as file:
        for i, c in enumerate(content):
            file.write(str(i) + seperator + str(c) + '\n')
    print('"' + filename + '" has been saved.')
