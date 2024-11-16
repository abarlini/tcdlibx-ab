import typing as tp
from typing import Annotated, Literal
import numpy as np
import numpy.typing as npt

Array3F = Annotated[npt.NDArray[np.float64], Literal[3]]
Array3x3F = Annotated[npt.NDArray[np.float64], Literal[3, 3]]
