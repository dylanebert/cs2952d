#!/usr/bin/env python3

class Expression:
    """
    This is the general structure of an expression for our arithmetic grammar.
    """
    def __init__(self, val):
        """
        :param val: the value of the expression
        """
        self.val = val

    def eval(self):
        """
        Evaluation of the expression
        """
        return self.val

    def __str__(self):
        """
        Nice representation
        """
        return str(self.val)

class Number(Expression):
    """
    The expression for numbers
    """
    def __init__(self, val):
        """
        Parameters: val -> int
        """
        super().__init__(val)

class Op1(Expression):
    """
    The unary operation. For the arithmetic grammar, this would be the negation.
    """
    def __init__(self, op, rhs):
        """
        :param op: string (~)
        :param rhs: Expression
        """
        super().__init__(op)
        self.rhs = rhs

    def eval(self):
        return {
            '~': -self.rhs.eval()
        }[self.val]

    def __str__(self):
        """
        String representation
        """
        return "({}, {})".format(self.val, str(self.rhs))

class Op2(Expression):
    """
    The binary operation. These would involve +,-,*
    """
    def __init__(self, op, lhs, rhs):
        """
        :param op: string
        :param lhs: Expression
        :param rhs: Expression
        """
        super().__init__(op)
        self.lhs = lhs
        self.rhs = rhs

    def eval(self):
        return {
            '+': self.lhs.eval() + self.rhs.eval(),
            '-': self.lhs.eval() - self.rhs.eval(),
            '*': self.lhs.eval() * self.rhs.eval()
        }[self.val]

    def __str__(self):
        """
        String representation
        """
        return "({}, {}, {})".format(self.val, str(self.lhs), str(self.rhs))
