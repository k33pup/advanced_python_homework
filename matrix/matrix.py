from tensor import Tensor
from typing import Union, Tuple, List, Any


class Matrix(Tensor):
    def __init__(self, dim: Tuple[int, int], data: List[Any]):
        if len(dim) != 2:
            raise ValueError("Matrix must be 2 dims!")
        super().__init__(dim, data)
        self.__rows_cnt, self.__columns_cnt = dim

    @property
    def rows_cnt(self) -> int:
        return self.__rows_cnt

    @property
    def columns_cnt(self) -> int:
        return self.__columns_cnt

    @property
    def data(self) -> List[Any]:
        return self._data

    def conv_rc2i(self, row: int, column: int) -> int:
        if not (0 <= row < self.rows_cnt and 0 <= column < self.__columns_cnt):
            raise ValueError("Wrong row or column index!")
        return row * self.__columns_cnt + column

    def conv_i2rc(self, index: int) -> Tuple[int, int]:
        if not (0 <= index < len(self.data)):
            raise ValueError("Wrong index!")
        return index // self.rows_cnt, index % self.rows_cnt

    @staticmethod
    def __convert_to_list(index: int | slice | list, size: int) -> List[int]:
        if isinstance(index, list):
            return index
        if isinstance(index, int):
            return [index] if index >= 0 else [index + size]
        if isinstance(index, slice):
            start, stop, step = index.indices(size)
            return list(range(start, stop, step))
        raise TypeError("Wrong index type!")

    def __getitem__(self, index: Union[int, tuple, list, slice]) -> Any:
        if not isinstance(index, int | list | slice | tuple):
            raise TypeError("Wrong index type!")
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("Wrong index type!")
            if isinstance(index[0], int) and isinstance(index[1], int):
                return self.data[self.conv_rc2i(index[0], index[1])]
            row_indexes = self.__convert_to_list(index[0], self.rows_cnt)
            column_indexes = self.__convert_to_list(index[1], self.columns_cnt)
        else:
            row_indexes = self.__convert_to_list(index, self.rows_cnt)
            column_indexes = self.__convert_to_list(slice(None), self.columns_cnt)
        result = []
        for row in row_indexes:
            result.extend(self.data[self.conv_rc2i(row, column)] for column in column_indexes)
        return Matrix((len(row_indexes), len(column_indexes)), result)

    def __str__(self) -> str:
        max_len = max(len(i.__str__()) for i in self.data)
        result = []
        for row in range(self.rows_cnt):
            row_str = "  ".join(f"{self.data[self.conv_rc2i(row, i)]:>{max_len}}" for i in range(self.__columns_cnt))
            result.append(f' {row_str}')
        return f"[\n{'\n\n'.join(result)}\n]"


if __name__ == "__main__":
    matrix = Matrix((10, 10), list(range(100)))

    print("1. M")
    print(matrix)
    print("\n2. M[1,1]")
    print(matrix[1, 1])
    print("\n3. M[1]")
    print(matrix[1])
    print("\n4. M[-1]")
    print(matrix[-1])
    print("\n5. M[1:4]")
    print(matrix[1:4])
    print("\n6. M[:4]")
    print(matrix[:4])
    print("\n7. M[4:]")
    print(matrix[4:])
    print("\n8. M[:]")
    print(matrix[:])
    print("\n9. M[1:7:2]")
    print(matrix[1:7:2])
    print("\n10. M[:, 1]")
    print(matrix[:, 1])
    print("\n11. M[1:4, 1:4]")
    print(matrix[1:4, 1:4])
    print("\n12. M[1:4, :4]")
    print(matrix[1:4, :4])
    print("\n13. M[1:4, 4:]")
    print(matrix[1:4, 4:])
    print("\n14. M[1:4, :]")
    print(matrix[1:4, :])
    print("\n15. M[-1:]")
    print(matrix[-1:])
    print("\n16. M[-2::-2]")
    print(matrix[-2::-2])
    print("\n17. M[-2::-2,1:4]")
    print(matrix[-2::-2, 1:4])
    print("\n18. M[:, :]")
    print(matrix[:, :])
    print("\n19. M[[1, 4]]")
    print(matrix[[1, 4]])
    print("\n20. M[:, [1,4]]")
    print(matrix[:, [1, 4]])
    print("\n21. M[[1, 4], [1, 4]]")
    print(matrix[[1, 4], [1, 4]])
