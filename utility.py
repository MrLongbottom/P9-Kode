from typing import List

import scipy.sparse as sp


def load_vector_file(filepath, separator=','):
    """
    Loads the content of a file as a dictionary
    :param filepath: path of file to be loaded. Should include folders and file type.
    :param separator: optional separator between values (default: ',')
    :return: dictionary containing the content of the file
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        dictionary = {}
        for line in file.readlines():
            kv = line.split(separator)
            value = kv[1].replace('\n', '')
            if len(value.split(' ')) > 1:
                value = value.split(' ')
            dictionary[int(kv[0])] = value
    return dictionary


def load_vecter_file_nonunique(filepath, separator=','):
    with open(filepath, 'r', encoding='utf-8') as file:
        listen = []
        for line in file.readlines():
            kv = line.split(separator)
            value = kv[1].replace('\n', '')
            if len(value.split(' ')) > 1:
                value = value.split(' ')
            listen.append((int(kv[0]), value))
    return listen


def rankify(dictionary):
    """
    convert dictionary of id:score to a ranked list of id's.
    :param dictionary:
    :return:
    """
    return list(dict(sorted(dictionary.items(), key=lambda x: x[1], reverse=True)).keys())


def save_vector_file(filepath, content, separator=','):
    """
    Saves content of list as a vector in a file, similar to a Word2Vec document.
    :param separator: separator between values
    :param filepath: path of file to save.
    :param content: list of content to save.
    :return: None
    """
    print('Saving file "' + filepath + '".')
    with open(filepath, "w", encoding='utf-8') as file:
        if isinstance(content, dict):
            for k, v in content.items():
                file.write(str(k) + separator + str(v) + '\n')
        else:
            for i, c in enumerate(content):
                file.write(str(i) + separator + str(c) + '\n')
    print('"' + filepath + '" has been saved.')


def save_vector_file_nonunique(filepath, content, separator=','):
    print('Saving file "' + filepath + '".')
    with open(filepath, "w", encoding='utf-8') as file:
        for i, c in content:
            file.write(str(i) + separator + str(c) + '\n')
    print('"' + filepath + '" has been saved.')


def slice_sparse_col(matrix: sp.csc_matrix, cols: List[int]):
    """
    Remove some columns from a sparse matrix.
    :param matrix: CSC matrix.
    :param cols: list of column numbers to be removed.
    :return: modified matrix without the specified columns.
    """
    cols.sort()
    ms = []
    prev = -1
    for c in cols:
        # add slices of the matrix, skipping column c
        ms.append(matrix[:, prev + 1:c - 1])
        prev = c
    ms.append(matrix[:, prev + 1:])
    # combine matrix slices
    return sp.hstack(ms)


def slice_sparse_row(matrix: sp.csr_matrix, rows: List[int]):
    """
    Remove some rows from a sparse matrix.
    :param matrix: CSR matrix.
    :param rows: list of row numbers to be removed.
    :return: modified matrix without the specified rows.
    """
    rows.sort()
    ms = []
    prev = -1
    for r in rows:
        # add slices of the matrix, skipping row r
        ms.append(matrix[prev + 1:r - 1, :])
        prev = r
    ms.append(matrix[prev + 1:, :])
    # combine matrix slices
    return sp.vstack(ms)
