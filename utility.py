

def load_vector_file(filepath, separator=','):
    """
    Loads the content of a file as a dictionary
    :param filepath: path of file to be loaded. Should include folders and file type.
    :param separator: optional separator between values (default: ',')
    :return: dictionary containing the content of the file
    """
    with open(filepath, 'r') as file:
        dictionary = {}
        for line in file.readlines():
            kv = line.split(separator)
            value = kv[1].replace('\n', '')
            if len(value.split(';')) > 1:
                value = value.split(';')
            dictionary[int(kv[0])] = value
    return dictionary


def rankify(dictionary):
    """
    convert dictionary of id:score to a ranked list of id's.
    :param dictionary:
    :return:
    """
    return list(dict(sorted(dictionary.items(), key=lambda x: x[1], reverse=True)).keys())



def rankify_topic(dictionary):
    return list(dict(sorted(dictionary.items(), key=lambda x: x[0], reverse=True)).keys())



def save_vector_file(filepath, content, separator=','):
    """
    Saves content of list as a vector in a file, similar to a Word2Vec document.
    :param separator: separator between values
    :param filepath: path of file to save.
    :param content: list of content to save.
    :return: None
    """
    print('Saving file "' + filepath + '".')
    with open(filepath, "w") as file:
        if isinstance(content, dict):
            for k, v in content.items():
                file.write(str(k) + separator + str(v) + '\n')
        else:
            for i, c in enumerate(content):
                file.write(str(i) + separator + str(c) + '\n')
    print('"' + filepath + '" has been saved.')
