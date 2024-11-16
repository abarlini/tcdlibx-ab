# -*- coding: utf-8 -*-
"""
VTK objects
"""
# import sys
# import os
import vtk
import typing as tp
import numpy as np

from tcdlibx.calc.cube_manip import CubeData, VecCubeData
from tcdlibx.graph.helpers import MyvtkActor
# from math import ceil

def dots2vtkarray(dots: np.ndarray) -> vtk.vtkPolyData:
    points1 = vtk.vtkPoints()
    for coords in dots:
        points1.InsertNextPoint(coords)

    pointPolyData1 = vtk.vtkPolyData()
    pointPolyData1.SetPoints(points1)
    return pointPolyData1

def fillcubeimage(data, vec=True, logscale=False, aslist=False):
    """
    Fills a vtkImageData object

    Arguments:
        data {dict} -- dictionary with a cube dataset
        logscale {bool} -- if True store the logaritm 

    Returns:
        vtkImageData
    """
    if vec and data.nval != 3:
        vec = False
    def setupcube(npts, loc2wrd, narray):
        vtkimage = vtk.vtkImageData()
        vtkimage.SetDimensions(*npts)
        vtkimage.SetOrigin(*loc2wrd[:3, 3])
        vtkimage.SetSpacing(*np.diag(loc2wrd[:3,:3]))
        vtkimage.AllocateScalars(vtk.VTK_DOUBLE, narray)
        return vtkimage

    if vec:
        cubeimage = setupcube(data.npts, data.loc2wrd, narray=4)
        vect = vtk.vtkDoubleArray()
        vect.SetNumberOfComponents(3)
        vect.SetNumberOfTuples(cubeimage.GetNumberOfPoints())
        vect.SetName('vector')
        norm = vtk.vtkDoubleArray()
        norm.SetNumberOfComponents(1)
        norm.SetNumberOfTuples(cubeimage.GetNumberOfPoints())
        norm.SetName('scalar')
    elif aslist:
        narray = data.nval
        scname = 'scalar'
        cubeimage = []
        for i in range(narray):
            cubeimage.append(setupcube(data.npts, data.loc2wrd, narray=1))
        arnpts = cubeimage[-1].GetNumberOfPoints()
    else:
        narray = data.nval
        scname = 'scalar{:02d}'
        cubeimage = setupcube(data.npts, data.loc2wrd, narray=narray)
        arnpts = cubeimage.GetNumberOfPoints()

    # NYI
    # cubeimage.AllocateScalars(vtk.VTK_DOUBLE, narray)
    # BUG only orthogonal grids
    # cubeimage.SetSpacing(*np.diag(data.loc2wrd[:3,:3]))
    if vec:
        scalar = np.sqrt(np.einsum("ij,ij->j", data.cube, data.cube))
        if logscale:
            scalar = np.log10(scalar+1e-13)
            scalar += 13
            scalar /= 1200
            # scalar[scalar < 3.1] = 0
        for i in range(data.npts[0]):
            for j in range(data.npts[1]):
                for k in range(data.npts[2]):
                    indices = (k * data.npts[1] + j) * data.npts[0] + i
                    indicesf = (i * data.npts[1] + j) * data.npts[2] + k
                    # norval = np.sqrt(np.dot(data.cube[:, indicesf], data.cube[:, indicesf]))
                    norval = scalar[indicesf]
                    norm.SetValue(indices, norval)
                    vect.SetTuple3(indices, *data.cube[:, indicesf])
        cubeimage.GetPointData().AddArray(vect)
        cubeimage.GetPointData().AddArray(norm)
    elif narray == 1:
        norm = vtk.vtkDoubleArray()
        norm.SetNumberOfComponents(1)
        norm.SetNumberOfTuples(arnpts)
        norm.SetName("scalar")
        for i in range(data.npts[0]):
            for j in range(data.npts[1]):
                for k in range(data.npts[2]):
                    indices = (k * data.npts[1] + j) * data.npts[0] + i
                    indicesf = (i * data.npts[1] + j) * data.npts[2] + k
                    norval = data.cube[indicesf]
                    norm.SetValue(indices, norval)
        cubeimage.GetPointData().AddArray(norm)
    else:
        for m in range(narray):
            norm = vtk.vtkDoubleArray()
            norm.SetNumberOfComponents(1)
            norm.SetNumberOfTuples(arnpts)
            norm.SetName(scname.format(m))
            for i in range(data.npts[0]):
                for j in range(data.npts[1]):
                    for k in range(data.npts[2]):
                        indices = (k * data.npts[1] + j) * data.npts[0] + i
                        indicesf = (i * data.npts[1] + j) * data.npts[2] + k
                        norval = data.cube[m, indicesf]
                        norm.SetValue(indices, norval)
            if aslist:
                cubeimage[m].GetPointData().AddArray(norm)
            else:
                cubeimage.GetPointData().AddArray(norm)
    # print(norm.GetRange())
    return cubeimage

def fillmolecule(atm, crd, wireframe=False,
                 opacity=1):
    """

    Args:
        moldata (_type_): _description_
    """
    mol = vtk.vtkMolecule()
    atoms = []
    for i in range(len(atm)):
        atoms.append(mol.AppendAtom(atm[i],
                                    *crd[i]))
    # try:
    #     bond = vtk.vtkPSimpleBondPerceiver()
    # except AttributeError:
    bond = vtk.vtkSimpleBondPerceiver()
    bond.SetInputData(mol)
    bond.Update()
    molout = bond.GetOutput()
    # Create a mapper
    # mapper = vtk.vtkPolyDataMapper()
    mapper = vtk.vtkMoleculeMapper()
    mapper.UseLiquoriceStickSettings()
    if wireframe:
        mapper.SetAtomicRadiusScaleFactor(0.03)
        mapper.SetBondRadius(0.03)
    # mapper.SetInputConnection(source.GetOutputPort())
    mapper.SetInputData(molout)
    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    return MyvtkActor(actor, mol)

def quiv3d(cubdata, lower=0.0001, upper=0.01, scale=1, logscale=False):
    """
    Return a list mlab.quiv3d
    param: cubedata a vectorial dataset as CubData object
    param: lower/upper the lower and upper bounds to filter
           the vector field
    param: logscale set if logaritmic
           no log lower=0.001, upper=0.2
           log lower=0.0082, upper=0.0092
    """

    _grid = fillcubeimage(cubdata, logscale=logscale)
    _grid.GetPointData().SetActiveVectors('vector')
    _grid.GetPointData().SetActiveScalars('scalar')
    # _bounds = _grid.GetScalarRange()
    # visualizzo 1/100?
    # _bounds2 = (0, _bounds[1]/100)

    arrow = vtk.vtkArrowSource()
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(_grid)
    glyphs.SetSourceConnection(arrow.GetOutputPort())
    # the mapper
    glyph_mapper =  vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.VisibilityOn()

    glyphs.SetVectorModeToUseVector()
    glyphs.SetScaleModeToScaleByScalar()
    # Scale factor
    glyphs.SetScaleFactor(scale)
    glyphs.SetColorModeToColorByScalar()
    # color map
    lut = vtk.vtkColorTransferFunction()
    lut.AddRGBPoint(lower, 1,0,0)
    lut.AddRGBPoint(upper, 0,1,0)
    glyph_mapper.SetLookupTable(lut)
    # filtering
    threshold = vtk.vtkThresholdPoints()
    threshold.SetInputData(_grid)
    threshold.ThresholdBetween(lower, upper)
    glyphs.SetInputConnection(threshold.GetOutputPort())

    return MyvtkActor(glyph_actor, glyphs)


def draw_nm3d(crd, evec, ian,
              cngsign=True,
              chgwght=True, scale=1,
              color="blue"):
    """_summary_

    Args:
        crd (_type_): _description_
        evec (_type_): _description_
        ian (_type_): _description_
        cngsign (bool, optional): _description_. Defaults to True.
        chgwght (bool, optional): _description_. Defaults to True.
        scale (int, optional): _description_. Defaults to 1.
        color (str, optional): _description_. Defaults to "blue".

    Returns:
        _type_: _description_
    """

    # weighted by the charge
    if chgwght:
        levec = evec * np.array(ian)[:, np.newaxis]
    norms = np.sqrt(np.einsum('ij,ij->i', levec, levec)).max()
    # normalized
    norm_evec = levec / norms
    # change the sign to be opposed to the TCD
    if cngsign:
        evec *= -1
    natm = int(crd.shape[0])
    # PolyData
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(natm)
    vect = vtk.vtkDoubleArray()
    vect.SetNumberOfComponents(3)
    vect.SetNumberOfTuples(natm)
    ones = vtk.vtkDoubleArray()
    ones.SetNumberOfComponents(1)
    ones.SetNumberOfTuples(natm)
    for i in range(natm):
        points.SetPoint(i, crd[i, :])
        ones.SetValue(i, 1.)
        vect.SetTuple3(i, *norm_evec[i, :])
    vect.SetName('vector')
    ones.SetName('ones')
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(vect)
    polydata.GetPointData().AddArray(ones)
    polydata.GetPointData().SetActiveVectors('vector')
    polydata.GetPointData().SetActiveScalars('ones')

    arrow = vtk.vtkArrowSource()
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(polydata)
    glyphs.SetSourceConnection(arrow.GetOutputPort())
    # the mapper
    glyph_mapper =  vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.VisibilityOn()

    glyphs.SetVectorModeToUseVector()
    glyphs.SetScaleModeToScaleByVector()
    # Scale factor
    glyphs.SetScaleFactor(scale)
    clrs = vtk.vtkNamedColors()
    glyphs.SetColorModeToColorByScalar()
    glyph_actor.GetProperty().SetColor(clrs.GetColor3d(color))

    return MyvtkActor(glyph_actor, glyphs)

def draw_vectors(crd, vecs, tps, scale=1):
    """_summary_

    Args:
        crd (_type_): _description_
        vecs (_type_): _description_
        tps (_type_): [1,2,3,4] 4 total electric
                      [-1,-2,-3,-4] magnetic
        scale (int, optional): _description_. Defaults to 1.
        color (str, optional): _description_. Defaults to "blue".

    Returns:
        _type_: _description_
    """

    # weighted by the charge
    # norms = np.sqrt(np.einsum('ij,ij->i', vecs, vecs)).max()
    norms = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))[0]
    # normalized
    norm_vecs = vecs / norms

    # multiply magnetic 1e3
    # norm_vecs[tps < 0,:]
    natm = int(crd.shape[0])
    # PolyData
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(natm)
    vect = vtk.vtkDoubleArray()
    vect.SetNumberOfComponents(3)
    vect.SetNumberOfTuples(natm)
    ones = vtk.vtkDoubleArray()
    ones.SetNumberOfComponents(1)
    ones.SetNumberOfTuples(natm)
    for i in range(natm):
        points.SetPoint(i, crd[i, :])
        ones.SetValue(i, tps[i])
        vect.SetTuple3(i, *norm_vecs[i, :])
    vect.SetName('vector')
    ones.SetName('ones')
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(vect)
    polydata.GetPointData().AddArray(ones)
    polydata.GetPointData().SetActiveVectors('vector')
    polydata.GetPointData().SetActiveScalars('ones')

    arrow = vtk.vtkArrowSource()
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(polydata)
    glyphs.SetSourceConnection(arrow.GetOutputPort())
    # the mapper
    glyph_mapper =  vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.VisibilityOn()

    # color map
    lut = vtk.vtkColorTransferFunction()
    lut.AddRGBPoint(-4, 1,.3,1)
    lut.AddRGBPoint(-3, 153/255,0,0)
    lut.AddRGBPoint(-2, 1,0,0)
    lut.AddRGBPoint(-1, 1,128/255,0)
    # lut.AddRGBPoint(0, 0,1,0)
    lut.AddRGBPoint(1, 125/255,1,0)
    lut.AddRGBPoint(2, 0,1,0)
    lut.AddRGBPoint(3, 0,102/255,0)
    lut.AddRGBPoint(4, 1,1,0)
    glyph_mapper.SetLookupTable(lut)

    glyphs.SetVectorModeToUseVector()
    glyphs.SetScaleModeToScaleByVector()
    # Scale factor
    glyphs.SetScaleFactor(scale)
    glyphs.SetColorModeToColorByScalar()

    return MyvtkActor(glyph_actor, glyphs)


def fillstreamline(cubdata: CubeData,
                   nseeds: tp.Optional[int] = 150,
                   center: tp.Optional[list] = [0., 0., 0.],
                   opacity: tp.Optional[float] = 0.3,
                   clipping: tp.Optional[tuple] = (1e2, 1e5),
                   minspeed: tp.Optional[tp.Union[float, None]] = None,
                   seeds: tp.Optional[tp.Union[np.ndarray, None]] = None,
                   scale_rad=1) -> MyvtkActor:
    """_summary_

    Args:
        cubdata (CubeData): _description_
        nseeds (tp.Optional[int], optional): _description_. Defaults to 150.
        center (tp.Optional[list], optional): _description_. Defaults to [0., 0., 0.].
        opacity (tp.Optional[float], optional): _description_. Defaults to 0.3.
        clipping (tp.Optional[tuple], optional): _description_. Defaults to (1e2, 1e5).
        minvel (tp.Optional[tp.Union[float, None]], optional): _description_. Defaults to None.
        scale_rad (int, optional): _description_. Defaults to 1.

    Returns:
        vtk.vtkActor: _description_
    """

    # Vectors stuff
    # https://stackoverflow.com/questions/57309203/plotting-vector-fields-efficiently-using-vtk-avoiding-excessive-looping
    _grid = fillcubeimage(cubdata)
    _grid.GetPointData().SetActiveVectors('vector')
    _grid.GetPointData().SetActiveScalars('scalar')
    _bounds = _grid.GetScalarRange()
    # visualizing only a portion between the two bounds
    _bounds2 = (_bounds[1]/clipping[1],
                _bounds[1]/clipping[0])
    # defining the seed points
    if seeds is None:
        _seeds = vtk.vtkPointSource()
        _seeds.SetCenter(*center)
        _seeds.SetNumberOfPoints(nseeds)
        _seeds.SetRadius(5.0)
        # possible different distributions, see at
        # https://vtk.org/doc/nightly/html/classvtkPointSource.html#a2029a3636eef7a32db31a10c9a904f9c
        # random distribution, seek how to get and save point positions
        _seeds.Update()
        _flag = True
    else:
        _seeds = dots2vtkarray(seeds)
        _flag = False

    # Streamlines stuff
    integrator=vtk.vtkRungeKutta45()
    streamline = vtk.vtkStreamTracer()
    streamline.SetInputData(_grid)
    if _flag:
        streamline.SetSourceConnection(_seeds.GetOutputPort())
    else:
        streamline.SetSourceData(_seeds)
    streamline.SetMaximumPropagation(50)
    streamline.SetIntegrator(integrator)
    streamline.SetInitialIntegrationStep(.1)
    streamline.SetIntegrationDirectionToBoth()
    streamline.SetComputeVorticity(True)
    if minspeed is None:
        minspeed = _bounds2[1] / 1e4
    streamline.SetTerminalSpeed(minspeed)
    # Building the tubes upone the streamlines
    streamtube = vtk.vtkTubeFilter()
    streamtube.SetInputConnection(streamline.GetOutputPort())
    # streamtube.SetInputArrayToProcess(3, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "norm")
    # streamtube.SetInputArrayToProcess(grid.GetPointData().GetNormals())
    streamtube.SetRadius(0.01)
    streamtube.SetNumberOfSides(12)
    # changes the radius according to the field magnitude
    streamtube.SetVaryRadiusToVaryRadiusByScalar()
    streamtube.CappingOn()
    streamtube.Update()
    # To change the colors
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.6, 0.0)
    lut.Build()

    streamline_mapper = vtk.vtkPolyDataMapper()
    streamline_mapper.SetInputConnection(streamtube.GetOutputPort())
    # streamline_mapper.SetColorModeToMapScalars()
    # streamline_mapper.SetColorModeToMapScalars()
    streamline_mapper.ScalarVisibilityOn()
    # streamline_mapper.SetScalarModeToUsePointFieldData()
    streamline_mapper.SetScalarRange(*_bounds2)
    # streamline_mapper.ColorByArrayComponent("norm", 0)
    # streamline_mapper.SetLookupTable(ctf)
    streamline_mapper.SetLookupTable(lut)
    # streamline_mapper.SelectColorArray('norm')
    streamline_actor = vtk.vtkActor()
    streamline_actor.SetMapper(streamline_mapper)
    streamline_actor.GetProperty().SetOpacity(opacity)
    streamline_actor.VisibilityOn()
    return MyvtkActor(streamline_actor, streamtube)

def countur(cubedata, isoval, active='scalar', colors=None, opacity=0.3):
    """
    Return vtk actor with the isosurface plotted
    param: xax, yax, zax 3d grid of dimension NxNxN
    val: values of the grid dots
    isoval: list of the wanted isovalue
    """
    assert len(isoval) < 3
    grid = fillcubeimage(cubedata)
    # grid.GetPointData().SetActiveVectors("vector")
    # grid.GetPointData().SetActiveScalars(active)
    return _countur(grid, isoval, active, colors, opacity)

def _countur(grid, isoval, active='scalar', colors=None, opacity=0.3):
    # grid.GetPointData().SetActiveVectors("vector")
    grid.GetPointData().SetActiveScalars(active)
    # bounds = grid.GetScalarRange()

    contourFilter = vtk.vtkContourFilter()
    contourFilter.SetInputData(grid)
    contourFilter.SetArrayComponent(0)
    # set isoval and define lut
    if colors is None or len(colors) < len(isoval):
        colors = [[1,0,0], # red
                 [0,0,1]] # blue
    elif isinstance(colors[0], str):
        clrs = vtk.vtkNamedColors()
        colors = [clrs.GetColor3d(clr) for clr in colors]
    # Add checks
    lut = vtk.vtkColorTransferFunction()
    for i in range(len(isoval)):
        contourFilter.SetValue(i, isoval[i])
        lut.AddRGBPoint(isoval[i], *colors[i])
    contourFilter.Update()
    # mapper
    isosurf_mapper = vtk.vtkPolyDataMapper()
    isosurf_mapper.SetInputConnection(contourFilter.GetOutputPort())
    isosurf_mapper.ScalarVisibilityOn()
    # colors
    isosurf_mapper.SetLookupTable(lut)
    # actor
    isosurf_actor = vtk.vtkActor()
    isosurf_actor.SetMapper(isosurf_mapper)
    isosurf_actor.GetProperty().SetOpacity(opacity)
    isosurf_actor.VisibilityOn()

    return MyvtkActor(isosurf_actor, contourFilter)


def draw_colorbar(targetactor: vtk.vtkActor, title: str,
                  nlabs: int = 5) -> MyvtkActor:
    """
    

    Args:
        targetactor (vtk.vtkActor): _description_
        title (str): _description_
        opacity (int, optional): _description_. Defaults to 1.
    """
    # Create a scalar bar
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(targetactor.GetMapper().GetLookupTable())
    scalarBar.SetTitle(title)
    scalarBar.SetNumberOfLabels(nlabs)
    # check these values
    scalarBar.SetMaximumWidthInPixels(100)
    scalarBar.SetMaximumHeightInPixels(400)
    scalarBar.SetPosition(0.1, 0.1)
    scalarBar.SetOrientationToVertical()
    scalarBar.GetTitleTextProperty().SetColor(0,0,0)
    scalarBar.GetTitleTextProperty().SetBold(True)

    return MyvtkActor(scalarBar, None)

def draw_cones_nogrid(cubdata: VecCubeData, point: np.ndarray, scale: float=0.1) -> MyvtkActor:
    npoints = point.shape[0]
    pcons = vtk.vtkPoints()
    pcons.SetNumberOfPoints(npoints)
    vect = vtk.vtkDoubleArray()
    vect.SetNumberOfComponents(3)
    vect.SetNumberOfTuples(npoints)
    ones = vtk.vtkDoubleArray()
    ones.SetNumberOfComponents(1)
    ones.SetNumberOfTuples(npoints)
    for i, val in enumerate(point):
        pcons.SetPoint(i, val)
        ones.SetValue(i, scale)
        vect.SetTuple3(i, *cubdata.get_value(val))
    vect.SetName('vector')
    ones.SetName('ones')
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pcons)
    polydata.GetPointData().AddArray(vect)
    polydata.GetPointData().AddArray(ones)
    polydata.GetPointData().SetActiveVectors('vector')
    polydata.GetPointData().SetActiveScalars('ones')
    cone = vtk.vtkConeSource()
    coneglyphs = vtk.vtkGlyph3D()
    coneglyphs.SetInputData(polydata)
    coneglyphs.SetSourceConnection(cone.GetOutputPort())
    coneglyph_mapper =  vtk.vtkPolyDataMapper()
    coneglyph_mapper.SetInputConnection(coneglyphs.GetOutputPort())
    coneglyph_actor = vtk.vtkActor()
    coneglyph_actor.SetMapper(coneglyph_mapper)
    coneglyph_actor.VisibilityOn()

    coneglyphs.SetVectorModeToUseVector()
    coneglyphs.SetScaleModeToScaleByScalar()
    return MyvtkActor(coneglyph_actor, coneglyphs)

def draw_ellipsoid(points: np.ndarray) -> MyvtkActor:
        vtkdots = dots2vtkarray(points)
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.05)
        glyphs = vtk.vtkGlyph3D()
        glyphs.SetInputData(vtkdots)
        glyphs.SetSourceConnection(sphere.GetOutputPort())
        # glyphs.Update()

        glyph_mapper =  vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.VisibilityOn()
        return MyvtkActor(glyph_actor, glyphs)



