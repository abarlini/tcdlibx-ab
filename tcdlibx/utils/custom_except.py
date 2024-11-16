#!/bin/python
"""
Customized Exception Class
"""

# Exceptions
class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class NoValidData(Error):
    """Exception raised for errors in the input DataFile.

    Attributes:
    expression -- input expression in which the error occurred
    message -- explanation of the error
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class NoValidArg(Error):
    """Exception raised for errors in the input DataFile.

    Attributes:
    expression -- input expression in which the error occurred
    message -- explanation of the error
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
