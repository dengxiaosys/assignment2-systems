import pytest

from cpp_add_demo import add


def test_adds_integers() -> None:
    assert add(2, 3) == 5
    assert isinstance(add(2, 3), int)


def test_adds_negative_integers() -> None:
    assert add(-7, 3) == -4


def test_accepts_keyword_arguments() -> None:
    assert add(a=7, b=8) == 15


def test_rejects_incompatible_types() -> None:
    with pytest.raises(TypeError):
        add(1.5, 2)


def test_rejects_int64_addition_overflow() -> None:
    with pytest.raises(OverflowError, match="int64 addition overflow"):
        add(2**63 - 1, 1)


def test_rejects_python_integer_outside_int64_range() -> None:
    with pytest.raises(TypeError):
        add(2**100, 1)
