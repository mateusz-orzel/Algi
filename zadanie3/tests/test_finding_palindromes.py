import pytest
from zadanie3.zad3 import Solution

@pytest.fixture
def sol():
    return Solution()

def test_non_palindrome(sol):
    result = sol.finding_palindromes("abc")
    assert not result

def test_palindrome_odd(sol):
    result = sol.finding_palindromes("level")
    assert result

def test_palindrome_even(sol):
    result = sol.finding_palindromes("deed")
    assert result

def test_empty_string(sol):
    result = sol.finding_palindromes("")
    assert result