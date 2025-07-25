# Script to check the dipole moments cube wrt the fchk file

import sys
import os
import argparse
from tcdlibx.calc.cube_manip import VecCubeData, CubeData, cube_parser


def open_cube(fname: str) -> CubeData:
    """Open the cube file and return the cube object."""
    if not os.path.exists(fname):
        raise FileNotFoundError(f'File {fname} not found.')
    cubdata = cube_parser(fname)
    return cubdata


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description='Check the dipole moments cube wrt the fchk file.')
    parser.add_argument('cub', type=str, help='cube file')
    parser.add_argument('--rotor', action="store_true", help='Compute the rotor of the cube file')
    return parser

def main():
    """Main function to check the dipole moments cube wrt the fchk file."""
    parser = build_parser()
    args = parser.parse_args()

    cub = open_cube(args.cub)

    if cub.nval == 3:
        cubedata = VecCubeData(cub)
    else:
        cubedata = cub

    if not isinstance(cubedata, VecCubeData) and args.rotor:
        raise TypeError("Cube data must be a VecCubeData object to compute the rotor.")

    print(f"Cube file: {args.cub}")
    # check only the electronic component
    cub_int = cubedata.integrate()
    if isinstance(cubedata, VecCubeData):
        print(f"cube int: {cub_int[0][0]:10.5f}{cub_int[0][1]:10.5f}{cub_int[0][2]:10.5f}")
        if args.rotor:
            cub_rot = cubedata.rotorintegrate()
            print(f"cube rotor: {cub_rot[0][0]:10.5f}{cub_rot[0][1]:10.5f}{cub_rot[0][2]:10.5f}")
    else:
        print(f"cube int: {cub_int:10.5f}")

    
if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        sys.exit(1)