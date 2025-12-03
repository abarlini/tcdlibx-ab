import unittest
import numpy as np

from tcdlibx.calc.cube_manip import CubeData, VecCubeData

try:
    from tcdlibx.graph import cube_graphvtk
    from tcdlibx.utils.vtk_utils import vtk
    VTK_AVAILABLE = True
except ModuleNotFoundError:
    cube_graphvtk = None

    class _Dummy:
        pass

    vtk = _Dummy()
    VTK_AVAILABLE = False


class TestMagnitudeMode(unittest.TestCase):
    def test_norm_and_volume_creation(self):
        if not VTK_AVAILABLE:
            self.skipTest("VTK is not installed; skipping volume rendering test")

        base = CubeData()
        base.npts = [2, 2, 2]
        base.nval = 3
        base.loc2wrd = np.identity(4)
        base.cube = np.ones((3, 8))

        vec_cube = VecCubeData(base)
        mag_cube = vec_cube.get_norm(cube=True)

        self.assertEqual(mag_cube.nval, 1)
        self.assertEqual(mag_cube.cube.shape, (8,))

        volume_actor = cube_graphvtk.volume_render_scalar(mag_cube, opacity_range=(0.1, 1.0))
        self.assertTrue(hasattr(volume_actor, "actor"))
        if hasattr(vtk, "vtkVolume"):
            self.assertIsInstance(volume_actor.actor, vtk.vtkVolume)


if __name__ == "__main__":
    unittest.main()
