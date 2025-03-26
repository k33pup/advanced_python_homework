from typing import Tuple, Any, List
from copy import deepcopy


class Tensor:
    def __init__(self, dim: Tuple[int, ...], data: List[Any]):
        if not self.__is_valid_input(dim, data):
            raise ValueError("Wrong data dimension!")
        self._dimensions = dim
        self._data = deepcopy(data)

    @staticmethod
    def __is_valid_input(dim: Tuple[int, ...], data: List[Any]) -> bool:
        if len(dim) == 0:
            return len(data) == 0
        result = dim[0]
        for i in range(1, len(dim)):
            result *= dim[i]
        return result == len(data)

    def __str__(self) -> str:
        if not self._data:
            return f"[\n]"
        max_len = max(len(i.__str__()) for i in self._data)
        return f'[\n  {"  ".join([f'{i.__str__():>{max_len}}' for i in self._data])}\n]'


if __name__ == "__main__":
    matr = Tensor((5, ), [1, 2, 3, 4, 6])
    print(matr)
