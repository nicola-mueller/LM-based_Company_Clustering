def save_strings_to_file(strings, filename):
    """
    This function saves a list of strings to a file with each string on a new line.

    :param strings: list of strings
    :param filename: name of the file
    """
    with open(filename, 'w') as f:
        for item in strings:
            f.write("%s\n" % item)
