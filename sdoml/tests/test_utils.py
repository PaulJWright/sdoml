from sdoml.utils.utils import get_minvalue, is_str_list, solve_list


def test_solve_list_1():
    assert solve_list(["a", "b", "c", "d"], ["c", "d", "a", "b"]) == [
        "c",
        "d",
        "a",
        "b",
    ]
    assert solve_list(["a", "b", "c", "d"], ["d", "b", "c", "a"]) == [
        "d",
        "b",
        "c",
        "a",
    ]


def test_min_list():
    assert get_minvalue([1, 2, 3, 4, 5]) == (1, 0)
    assert get_minvalue([2, 1, 3, 4, 5]) == (1, 1)
    assert get_minvalue([6, 2, 3, 4, 5]) == (2, 1)
    assert get_minvalue([6, 4, 4, 3, 5]) == (3, 3)


def test_str_list():
    assert is_str_list(["a", "b", "c"]) is True
    assert is_str_list(["a", 2, "c"]) is False
    assert is_str_list([1, 2, 2]) is False
    assert is_str_list([1, 2.3, 2]) is False
    assert is_str_list([True, False, False]) is False
