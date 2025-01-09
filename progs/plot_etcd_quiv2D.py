#!/bin/python3
"""
FIXME old script to be updated
Es: python3 plot_ETCD_quiv2D.py -s '2+9' -u 300 -g 1 -a y --vscale=100\
        --printNM allene_F2.anh.2nq.GVPT2.full.fchk\
        TCD:TCD/allenef2_TCD_SXXX.cube\
        NAC:new_phase/allene_F2_SXXX.anh.2nq.GVPT2.full.log\
        EXC:excstate/allene_F2_NACXXX.fchk
"""
import sys
import os
import re
import argparse
from math import ceil, floor, pi
import numpy as np
if "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import custom library

from estampes.data.physics import PHYSFACT, PHYSCNST, phys_fact

import tcdlibx.calc.cube_manip as cb
import tcdlibx.graph.cube_graphic as cbplt
from tcdlibx.utils.custom_except import NoValidData
from tcdlibx.utils.color_out import Colors
import tcdlibx.io.fchk_io as fio
from tcdlibx.utils.var_tools import get_ivib, print_lines
# import variabiles
from tcdlibx.utils.mol_data import ELEMENTS, AT_COL2, AT_RAD
try:
   #  '''
   #  Specific connectivity matrix can be provided
   #  as molbonds.py in the cuccent directory formatted as:
   #      AT_BONDS = [ (at_1, at_2),
   #                   (at_3, at_5),
   #                   etc..]
   #  '''
    from molbonds import AT_BONDS
except ImportError:
    AT_BONDS = None


PROGNAME = os.path.basename(sys.argv[0])
NSTEP_BOX = 1
# SCALE_ARROW = 1.0e9
SCALE_ARROW = None
PRE_FACT = 1


def build_parser():
    """
    Build options parser.
    """
    par = argparse.ArgumentParser(prog=PROGNAME,
                                  formatter_class=argparse.RawTextHelpFormatter)
    # MANDATORY ARGUMENTS
    txt = "Gaussian fchk with data on state of interest"
    par.add_argument('refstate', help=txt)
    txt = """Various name templates for data of interest.
The part to replace with the electronic state index are indicated with "X"'s
Example: cube_SXX.cube
In order:
- transition current density (TCD)
- non-adiabatic couplings (NAC)
- excited-states electronic data (EXC)"""
    par.add_argument('tmplfiles', nargs='+', help=txt)
    # OPTIONAL ARGUMENTS
    par.add_argument('-p', '--print', action='store_true',
                     help='Print molecule')
    # -- ELECTRONIC STATE OPTIONS
    state = par.add_argument_group('Electronic state(s) selection options')
    state.add_argument('-l', '--lower', type=int, default=1,
                       help='Lowest electronic state to include in sum')
    state.add_argument('-u', '--upper', type=int, default=3,
                       help='Upper electronic state to include in sum')
    state.add_argument('-1', '--only', type=int,
                       help='Electronic state to include (only 1)')
    state.add_argument('-m', '--multi',
                       help='Choose multiple states (as comma-separated list)')
    state.add_argument('--noNAC', action="store_true",
                       help='Deactivates the NAC weight.')
    state.add_argument('--noweight', action="store_true",
                       help='Deactivates the weight (both NAC and transition energy).')

    # -- VIBRATIONAL STATES
    vib = par.add_argument_group('Vibrational state/mode selection options')
    vib.add_argument('-s', '--vibstate', default='1',
                     help='Vibrational state of interest')
    # -- DRAWING PARAMETERS
    # vis = p.add_subparsers('-2d',help='2D-projection')

    draw = par.add_argument_group('Drawing parameters')
    # raw.add_argument('-v', '--view', choices=('2d', '3d'), default='3d',
    #                 help='type of graph. 2D or 3D')
    draw.add_argument('-a', '--axis', choices=('x', 'y', 'z'),
                      help='Axis for the projection')
    draw.add_argument('--vscale', type=float, default=5.,
                      help='Overall scaling factor for the arrows')
    draw.add_argument('-g', '--grid', type=int,
                      help='Sets the step for the grid construction')
    draw.add_argument('--xmin', type=float,
                      help='Lower bound along x for the current vectors')
    draw.add_argument('--xmax', type=float,
                      help='Upper bound along x for the current vectors')
    draw.add_argument('--ymin', type=float,
                      help='Lower bound along y for the current vectors')
    draw.add_argument('--ymax', type=float,
                      help='Upper bound along y for the current vectors')
    draw.add_argument('--zmin', type=float,
                      help='Lower bound along z for the current vectors')
    draw.add_argument('--zmax', type=float,
                      help='Upper bound along z for the current vectors')
    draw.add_argument('--printNM', action='store_true',
                      help='Print normal modes')
    draw.add_argument('--printVec', action='store_true',
                      help='print EDTM and MDTM')
    draw.add_argument('--notitle', action='store_true',
                      help='do not print title')
    draw.add_argument('--saveCube', action='store_true',
                      help='Save the last computed cube')
    draw.add_argument('--saveSignedCube', action='store_true',
                      help='Save the last computed cube')
    draw.add_argument('--setlimit',
                      help='Set the axis limits for a 2d plot as \
                      comma separated values: [xmin,xmax,ymix,ymax]')
    draw.add_argument('--symplot', action='store_true',
                      help='''Center the 2D plot at the Origin (0,0)''')
    draw.add_argument('--printConv', action='store_true',
                      help='''Save in file the ETDM and MTDM for each state''')

    return par


if __name__ == '__main__':
    cp = Colors()
    DEBUG = True
    PARSER = build_parser()
    OPTS = PARSER.parse_args()
    # Scale to use print_mol2d
    SCF = [1., 1.]
    # Check that reference state data file exists
    if not OPTS.refstate:
        print(cp.printred('ERROR: refstate does not exist'))
    # Analys template files
    ntmpl = len(OPTS.tmplfiles)
    tmplfiles = {}
    for item in OPTS.tmplfiles:
        if item.startswith('TCD:'):
            tmplfiles['TCD'] = item[4:]
        elif item.startswith('NAC:'):
            tmplfiles['NAC'] = item[4:]
        elif item.startswith('EXC:'):
            tmplfiles['EXC'] = item[4:]
    if len(tmplfiles) == 0:
        tmplfiles['TCD'] = OPTS.tmplfiles[0]
        if ntmpl > 1:
            tmplfiles['NAC'] = OPTS.tmplfiles[1]
        else:
            tmplfiles['NAC'] = tmplfiles['TCD']
        if ntmpl > 2:
            tmplfiles['EXC'] = OPTS.tmplfiles[2]
        else:
            tmplfiles['EXC'] = tmplfiles['NAC']
    elif len(tmplfiles) != ntmpl:
        print('ERROR: Cannot mix keyword based and position templates')
        sys.exit()
    # elif len(tmplefiles) == 1:
    #     key = tmplfiles.keys()[0]
    #     for item in ('NAC', 'EXC', 'TCD'):
    #         if item not in tmplfiles:
    #             tmplfiles[item] = tmplfiles[key]
    else:
        if 'NAC' not in tmplfiles or 'TCD' not in tmplfiles:
            print('Provide at least template for NAC and TCD')
        if 'EXC' not in tmplfiles:
            tmplfiles['EXC'] = tmplfiles['NAC']

    tmplexts = {}
    for key in tmplfiles:
        tmplexts[key] = os.path.splitext(tmplfiles[key])[1][1:]
    tmplftypes = {}
    tmplfmt = {}
    for key in tmplfiles:
        # Analyze pattern
        fext = tmplexts[key]
        if key == 'TCD':
            if fext in ('cub', 'cube'):
                tmplftypes[key] = 'cube'
            else:
                fmt = 'ERROR: Cube file extension expected but "{}" found'
                print(cp.printred(fmt.format(fext)))
                sys.exit()
        elif key in ('NAC', 'EXC'):
            if fext in ('fch', 'fchk'):
                tmplftypes[key] = 'fchk'
            elif fext in ('log', 'out'):
                tmplftypes[key] = 'log'
            elif fext in ('dat'):
                tmplftypes[key] = 'dat'
            else:
                fmt = 'ERROR: Formatted chk/output file extension expected but "{}" found'
                print(cp.printred(fmt.format(fext)))
                sys.exit()
        res = re.findall('(X+)', tmplfiles[key])
        if not tmplftypes[key] == 'dat':
            if not res:
                print(cp.printred("ERROR: Missing X's in filename pattern"))
                sys.exit()
            if len(res) > 1:
                print(cp.printred("ERROR: More than 1 sequence of X's found."))
                sys.exit()
            tmplfmt[key] = tmplfiles[key].replace(res[0], '{{:0{:d}d}}'.format(len(res[0])))
        else:
            tmplfmt[key] = tmplfiles[key]

    if OPTS.only:
        minels = OPTS.only
        maxels = OPTS.only
    else:
        minels = OPTS.lower
        maxels = OPTS.upper
    if minels > maxels:
        print(cp.printorange('WARNING: Electronic state bounds appear inverted. Correcting.'))
        minels, maxels = maxels, minels
    #ivib = OPTS.vibstate - 1
    if OPTS.grid:
        ngrdstp = OPTS.grid
    else:
        ngrdstp = NSTEP_BOX
    #if not OPTS.axis:
        #print("Error: Axis not specified")
        #sys.exit()
    # Get molecule information
    data = fio.fchk_vib_parser(OPTS.refstate)
    if OPTS.print:
        print('''
########################################
###          MOLECULAR DATA          ###
########################################

Coordinates (in Bohr)
''')
        fmt = 'Atom {:3d}  {:2s}  {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}'
        for ia in range(data['natoms']):
            ian = data['ian'][ia]
            xyz = data['crd'][ia, :] / PHYSFACT.bohr2ang
            print(fmt.format(ia+1, ELEMENTS[ian], c=xyz))
    nvib = data['natoms']*3 - 6
    evec = data['evec']

    lvibosc = OPTS.vibstate.split('+')
    dmodes = {}
    nquanta = 0
    for vibosc in lvibosc:
        if '*' in vibosc:
            data = vibosc.split('*')
            numq = int(data[0])
            mode = int(data[1])
        else:
            numq = 1
            mode = int(vibosc)
        nquanta += numq
        if mode > nvib:
            print('ERROR: Absolute variational state not yet available')
            sys.exit()
        if mode in dmodes:
            dmodes[mode] += numq
        else:
            dmodes[mode] = numq
    smodes = []
    lmodes = sorted(dmodes.keys(), reverse=True)
    for mode in lmodes:
        smodes.append('{}({})'.format(mode, dmodes[mode]))
    idvstate = get_ivib(nquanta, lmodes, dmodes, nvib)

    # Use 1st electronic state to build box
    title = 'CURRENT DENSITY FOR MODE {}'.format(OPTS.vibstate)
    print('''
########################################
### {:^32s} ###
########################################
'''.format(title))

    if OPTS.axis:
        resfolder = os.path.join("Q2D_plot")
        try:
            os.stat(resfolder)
        except OSError:
            os.mkdir(resfolder)
        ressubfolder = os.path.join(resfolder, "mode{:02d}{}".format((idvstate), OPTS.axis))
        try:
            os.mkdir(ressubfolder)
        except OSError:
            bakname = ressubfolder+'.back'
            counter = 0
            while os.path.exists(bakname):
                counter += 1
                bakname = ressubfolder+'.back'+str(counter)
            print(cp.printorange("WARNING: {} already exists: moved to {}".format(ressubfolder,bakname)))
            os.renames(ressubfolder, bakname)
            os.mkdir(ressubfolder)

    # if anharm
    # try:
    #     all_nmodes = get_anha_nm(OPTS.refstate)
    # except NoValidData as err:
    #     all_nmodes = evec

    if OPTS.printConv:
        elc = []
        mag = []
        lines = []
    #DEBUG
    #ofile = open("test.csv",'w')
    #ofile.write("\t Mu \t Mag \t\n")
    if tmplftypes['NAC'] == 'dat':
        NACS = np.loadtxt(tmplfmt['NAC'])


    flag = 0
    if OPTS.multi:
        lst_elst = [int(s) for s in OPTS.multi.split(',')]
    else:
        lst_elst = range(minels, maxels+1)

    for elst in lst_elst:
        fnames = {}
        for key in tmplfiles:
            fnames[key] = tmplfmt[key].format(elst)
            if not os.path.exists(fnames[key]):
                fmt = 'NOTE: Data files for the state num. {} missing. Skipping.'
                print(fmt.format(elst))
                if elst > maxels and flag:
                    print('ERROR: Not enough data file. Quitting.')
                    sys.exit()
                fnames = False
                continue
        if not fnames:
            continue


        print('Including electronic state num. {}'.format(elst))

        if OPTS.noNAC or OPTS.noweight:
            NACvib = 1.0
        else:
            if tmplftypes['NAC'] == 'fchk':
                dNAC = fio.get_nac(fnames['NAC'])
                all_nmodes = evec
                freq = data['frq']
                # del tmp
                PRE_FACT = 2 * np.sqrt(freq[int(idvstate-1)] /
                                       (phys_fact('au2cm-1') *
                                       PHYSCNST.finestruct) * pi)
            elif tmplftypes['NAC'] == 'log':
                dNAC = fio.get_nac(fnames['NAC'], smodes)
                all_nmodes = fio.get_anha_nm(OPTS.refstate)
                if not dNAC and nquanta == 1:
                    dNAC = fio.get_nac(fnames['NAC'])
            else:
                dNAC = NACS[elst - 1]
            if dNAC is False:
                print('ERROR: Non-adiabatic couplings not found. Exiting')
                sys.exit()
            else:
                if isinstance(dNAC, list):
                    # FIXME to check
                    NACvib = np.dot(evec, np.array(dNAC))[int(idvstate-1)]
                    print('state:{} NAC:{}'.format(elst, NACvib))
                else:
                    NACvib = dNAC

        if OPTS.noweight:
            X = 1.0
        else:
            Etrans = fio.get_transition_energy(fnames['EXC'])
            X = NACvib/Etrans
        if not flag:
            cubdat = cb.cube_parser(fnames['TCD'])
            cubdat.make_box()
            cubdat *= X
            # flag = 0

        else:
            cubdat += cb.cube_parser(fnames['TCD']) * X
        if OPTS.xmin or OPTS.xmax or OPTS.ymin or \
           OPTS.ymax or OPTS.zmin or OPTS.zmax:
            vec = cbplt.set_subgrid(cubdat,
                                    OPTS.xmin, OPTS.xmax, OPTS.ymin,
                                    OPTS.ymax, OPTS.zmin, OPTS.zmax)

        # DEBUG
        if not(OPTS.noNAC or OPTS.noweight) \
        and (OPTS.printConv or ((elst == lst_elst[-1])
                                and (OPTS.printVec or
                                     OPTS.saveSignedCube))):
            mu_state = cb.mu_integrate(cubdat)
            mg_state = cb.mag_integrate(cubdat)
            if OPTS.printVec and (elst == lst_elst[-1]):
                dcube = cb.CubeData()
                dcube.natoms = 0
                dcube.origin = np.array([0., 0., 0.])
                dcube.npts = [2, 2, 2]
                dcube.nval = 3
                dcube.step = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
                dcube.cube = np.zeros((3, 8))
                dcube.cube[:, 0] = mu_state
                outfile = os.path.splitext(OPTS.refstate)[0] + '_EDTM{}_U{}.cube'.format(OPTS.vibstate, lst_elst[-1])
                comment = 'EDTM vibmodes: {} Maxstates {}'.format(OPTS.vibstate, lst_elst[-1])
                cb.print_cube(dcube, fname=outfile, comment=comment)
                dcube.cube[:, 0] = mg_state
                outfile = os.path.splitext(OPTS.refstate)[0] + '_MDTM{}_U{}.cube'.format(OPTS.vibstate, lst_elst[-1])
                comment = 'MDTM vibmodes: {} Max states: {}'.format(OPTS.vibstate, lst_elst[-1])
                cb.print_cube(dcube, fname=outfile, comment=comment)
            if OPTS.printConv:
                norm_mu = np.linalg.norm(mu_state)
                norm_mg = np.linalg.norm(mg_state)
                dot_mumg = np.dot(mu_state, mg_state)
                elc.append(norm_mu)
                mag.append(norm_mg)

                cola = cp.printgreen('{:12.4E}'.format(norm_mu))
                if not flag:
                    print('Electronic EDTM: {}'.format(cola))
                    error = 0.
                else:
                    error = abs(elc[flag]-elc[flag-1])/elc[flag-1]
                    print('Electronic EDTM: {} Variation: {:12.4E}'.format(cola, error))

                lintpl = '{:4d} {a[0]:12.4E} {a[1]:12.4E} {a[2]:12.4E} {b[0]:12.4E} {b[1]:12.4E} {b[2]:12.4E}\n'
                lines.append(lintpl.format(elst, a=mu_state, b=mg_state))

                if elst == lst_elst[-1]:
                    # BUG: print in the same folder
                    print_lines(lines, elst, idvstate, '.')

        #BUG if skips the last cube
        if OPTS.axis and (elst == lst_elst[-1]):
            if not os.path.exists(os.path.join(ressubfolder,
                                               'quiv2D_{:03d}.pdf'.format(elst))):
                vec2, box2 = cbplt.simp_proj(cubdat, OPTS.axis)
                fig0, ax0 = plt.subplots()
                if not OPTS.notitle:
                    ax0.set_title('State: {:03d}'.format(elst))
                if OPTS.setlimit:
                    tmp = [float(s) for s in OPTS.setlimit[1:-1].split(',')]
                elif OPTS.symplot:
                    val1 = abs(max(ceil(10 * box2[0, 0]),
                                   floor(10 * box2[0, -1])))/10
                    val2 = abs(max(ceil(10 * box2[1, 0]),
                                   floor(10 * box2[1, -1])))/10
                    tmp = [-val1, val1, -val2, val2]
                else:
                    tmp = [box2[0, 0], box2[0, -1], box2[1, 0], box2[1, -1]]
                ax0.set_xlim(tmp[0:2])
                ax0.set_ylim(tmp[2:4])
                ax0.set_xlabel('Bohr')
                ax0.set_ylabel('Bohr')
                cbplt.draw_mol2d(ax0, cubdat.crd, cubdat.ian, OPTS.axis, SCF,
                                 conmat=AT_BONDS, to_bohr=True)
                if OPTS.printNM:
                    # FIXME: the 1 is the scaling factor OPTS.scaleNM
                    cbplt.draw_nm2d(ax0, data['crd'], all_nmodes[int(idvstate-1)],
                                    OPTS.axis, 1., to_bohr=True)
                    # get colormap
                    colma = plt.cm.get_cmap("seismic")
                    quiv = ax0.quiver(box2[0, :], box2[1, :], vec2[0, :],
                                      vec2[1, :], units='width',
                                      scale=OPTS.vscale, cmap=colma,
                                      zorder=3)
                fig0.savefig(os.path.join(ressubfolder,
                                          'quiv2D_{:03d}.pdf'.format(elst)))
                plt.close()
        flag += 1

    if OPTS.saveCube:
        cubdat.cube *= PRE_FACT
        outfile = os.path.splitext(OPTS.refstate)[0] + '_VTCD{}_U{}.cube'.format(OPTS.vibstate, lst_elst[-1])
        comment = 'VTCD Nmodes: {} Last State: {}'.format(OPTS.vibstate, lst_elst[-1])
        cb.print_cube(cubdat, fname=outfile, comment=comment)

    if OPTS.saveSignedCube:
        VEC1, VEC2 = cb.mask_cube(cubdat, mu_state)
        outfile = os.path.splitext(OPTS.refstate)[0] + '_VTCD{}_{}.cube'.format(OPTS.vibstate, 'Plus')
        comment = 'VTCD Nmodes: {} Signed: {}'.format(OPTS.vibstate, 'Plus')
        cb.print_cube(cubdat, fname=outfile, comment=comment, vec_pr=VEC1)
        outfile = os.path.splitext(OPTS.refstate)[0] + '_VTCD{}_{}.cube'.format(OPTS.vibstate, 'Minus')
        comment = 'VTCD Nmodes: {} Signed: {}'.format(OPTS.vibstate, 'Minus')
        cb.print_cube(cubdat, fname=outfile, comment=comment, vec_pr=VEC2)


    if not (OPTS.axis or OPTS.saveCube or OPTS.printConv):
        VMAX = np.abs(cubdat.cube).max(1)
        if not SCALE_ARROW:
            if OPTS.vscale:
                x_cmp = OPTS.vscale
            else:
                x_cmp = 2.
            vscale = x_cmp/VMAX.max()
        else:
            vscale = SCALE_ARROW
        fmt = 'VXmax = {v[0]:12.6e}, VYmax = {v[1]:12.6e},'\
              + ' VZmax = {v[2]:12.6e}, Scale = {s:12.6e}'
        print(fmt.format(v=VMAX, s=vscale))
        fig = plt.figure()
        fig.clf()
        ax = fig.gca(projection='3d')
        ax.set_xlim(cubdat.box[0, 0], cubdat.box[0, -1])
        ax.set_ylim(cubdat.box[1, 0], cubdat.box[1, -1])
        ax.set_zlim(cubdat.box[2, 0], cubdat.box[2, -1])
        cbplt.draw_cube(ax, cubdat, lvec=vscale)
        cbplt.draw_mol(ax, cubdat.crd, cubdat.ian,
                       conmat=AT_BONDS, to_bohr=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='best')
        # DEBUG
        # ofile.close()

        plt.show()

    #if not (flag == 0) and OPTS.axis:
    #    if os.path.exists(os.path.join(ressubfolder,'combined_q2d.pdf')):
    #        os.remove(os.path.join(ressubfolder,'combined_q2d.pdf'))
    #    bashCommand1 = "pdfunite quiv2D_*.pdf combined_q2d.pdf"
    #    bashCommand2 = "rm quiv2D_*.pdf"
    #    process1 = subprocess.check_output(['bash','-c', bashCommand1], cwd=ressubfolder)
    #    process2 = subprocess.check_output(['bash','-c', bashCommand2], cwd=ressubfolder)
