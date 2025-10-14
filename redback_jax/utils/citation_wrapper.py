def citation_wrapper(r):
    """
    Wrapper for citation function to allow functions to have a citation attribute
    :param r: proxy argument
    :return: wrapped function
    """

    def wrapper(f):
        f.citation = r
        return f

    return wrapper