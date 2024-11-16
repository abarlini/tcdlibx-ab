#!/url/local/bin/python
import sys
import copy
from tcdlibx.calc.cube_manip import cube_parser, print_cube

AT_NUM = int(sys.argv[1])

for i in range(1, AT_NUM+1, 1):
    tmp = cube_parser("basin{:04d}.cube".format(i))
    if i == 1:
        res = copy.deepcopy(tmp)
    else:
        tmp.cube[tmp.cube == 1] = i
        res += tmp

print_cube(res, 'partition', AT_NUM)
