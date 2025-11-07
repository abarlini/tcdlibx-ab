# Script to check the dipole moments cube wrt the fchk file

import sys
#import numpy as np
import os
import argparse
import re
from estampes.data.physics import phys_fact
from tcdlibx.calc.cube_manip import VtcdData, CubeData, cube_parser
from tcdlibx.graph.helpers import VibMolecule
from tcdlibx.io.estp_io import get_vibmol
#from tcdlibx.utils.conversion_units import edip_cgs, mdip_cgs, ele_mdip_cgs, ele_edip_cgs

# FIXME: check the exceptions? leave it to the gui? 
def open_fchk(fname: str) -> VibMolecule:
    """Open the fchk file and return the molecule object."""
    if not os.path.exists(fname):
        raise FileNotFoundError(f'File {fname} not found.')
    fchk = VibMolecule(get_vibmol(fname))
        # Fix properly the exceptions
        # except Exception as err: print(err)

    # except Exception as err: print(err)
    return fchk


def open_cube(fname: str, legacy: bool) -> CubeData:
    """Open the cube file and return the cube object."""
    if not os.path.exists(fname):
        raise FileNotFoundError(f'File {fname} not found.')
    elements = True
    if legacy:
        elements = False
    cubdata = cube_parser(fname, elements)
    return cubdata


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description='Check the dipole moments cube wrt the fchk file.')
    parser.add_argument('fchk', type=str, help='fchk file')
    parser.add_argument('cub', type=str, help='cube template name')
    parser.add_argument('--fout', type=str, default="vcd_test.txt", help='output filename')
    parser.add_argument('--legacy', action="store_true", help='sqrt factor for old cube')
    return parser

def main():
    """Main function to check the dipole moments cube wrt the fchk file."""
    parser = build_parser()
    args = parser.parse_args()
    fchk = open_fchk(args.fchk)
    tmplname = args.cub
    fout = args.fout
    res = re.findall('(X+)', tmplname) 
    if not res:
        print("ERROR: Missing X's in filename pattern")
        sys.exit()
    if len(res) > 1:
        print("ERROR: More than 1 sequence of X's found.")
        sys.exit()
    tmplfmt = tmplname.replace(res[0], '{{:0{:d}d}}'.format(len(res[0])))
    
    # Write header to output file before starting calculations
    with open(fout, 'w') as f:
        f.write("DTM Analysis Results\n")
        f.write("=" * 131 + "\n")
        header = (
            f"N. {'Freq.':^8s}"
            f"{'MFP (electric)':^30s}"
            f"{'MFP (magnetic)':^30s}"
            f"{'TCD_tot (electric)':^30s}"
            f"{'TCD_tot (magnetic)':^30s}\n"
        )
        f.write(header)
        f.write("-" * 131 + "\n")
    
    for i in range(1, fchk.ntrans + 1):
        fname = tmplfmt.format(i)
        if not os.path.exists(fname):
            print(f"ERROR: File {fname} not found.")
            sys.exit()
        tmp_cube = open_cube(fname, legacy=args.legacy)
        tcddata = VtcdData(tmp_cube,
                           fchk._moldata['evec'][i-1],
                           fchk._moldata['freq'][i-1])
        fchk.add_tcd(i-1, tcddata)
        nuc_cntr = fchk.get_dtm(i-1, tps='nuc', cgs=False)
        mfp_dtm = fchk.get_dtm(i-1, tps='tot', cgs=False)
        tcd_dtm = fchk.get_tcd_dtm(i-1, cgs=False)
        
        vib_length = fchk._moldata['freq'][i-1] / phys_fact("au2cm1")
        tcd_electric = tcd_dtm[0] / vib_length
        tcd_tot = (nuc_cntr[0] + tcd_electric, nuc_cntr[1] + tcd_dtm[1])
        
        # Print results for each state to output file
        with open(fout, 'a') as f:
            line = (
                f"{i:3d}: "
                f"{fchk._moldata['freq'][i-1]:8.2f} cm-1, "
                f"{mfp_dtm[0][0]:10.5f} {mfp_dtm[0][1]:10.5f} {mfp_dtm[0][2]:8.4f}  "
                f"{mfp_dtm[1][0]:10.5f} {mfp_dtm[1][1]:10.5f} {mfp_dtm[1][2]:8.4f}  "
                f"{tcd_tot[0][0]:10.5f} {tcd_tot[0][1]:10.5f} {tcd_tot[0][2]:8.4f}  "
                f"{tcd_tot[1][0]:10.5f} {tcd_tot[1][1]:10.5f} {tcd_tot[1][2]:10.5f}\n"
            )
            f.write(line)
        fchk.remove_tcd(i-1)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        sys.exit(1)
