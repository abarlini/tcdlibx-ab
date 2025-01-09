import os
import typing as tp
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
from PySide6.QtCore import QRegularExpression, QLocale
from PySide6.QtGui import QRegularExpressionValidator, QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QCheckBox
from PySide6.QtWidgets import QLineEdit, QFileDialog, QGridLayout, QPushButton
from tcdlibx.utils.var_tools import fuzzy_equal

class SavePngDialog(QDialog):
    def __init__(self, parent=None, fname="tcdfigure.png"):
        super().__init__(parent)
        self._fname = fname
        self._okexit = False
        nameLabel = QLabel('png name:', self)
        self.pngname = QLineEdit(self)
        png_valid = QRegularExpressionValidator()
        # regexp = QRegularExpression()
        # FIXME improve the regexp
        pattern = r'.*\.png'
        regexp = QRegularExpression(pattern)
        png_valid.setRegularExpression(regexp)
        # png_valid.setLocale(QLocale('English'))
        self.pngname.setValidator(png_valid)
        self.pngname.setText(self._fname)
        self.pngname.textEdited.connect(self._setcheck)

        self.setWindowTitle("Save PNG file")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.hlay = QHBoxLayout()
        self.hlay.addWidget(nameLabel)
        self.hlay.addWidget(self.pngname)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.okbtn = self.buttonBox.button(QDialogButtonBox.Ok)

        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._getpng)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay = QVBoxLayout()
        self.vlay.addLayout(self.hlay)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def _getpng(self):
        self._fname = self.pngname.text()
        self._okexit = True

    def _setcheck(self):
        if self.pngname.hasAcceptableInput():
            self.okbtn.setEnabled(True)
        else:
            self.okbtn.setEnabled(False)
            #self.buttonBox.


class SavePngSeriesDialog(QDialog):
    def __init__(self, parent=None, fname="tcd-gif-XXX.png"):
        super().__init__(parent)
        self._fname = fname
        self._okexit = False
        nameLabel = QLabel('template name for png name:', self)
        self.pngname = QLineEdit(self)
        png_valid = QRegularExpressionValidator()
        # regexp = QRegularExpression()
        # FIXME improve the regexp
        pattern = r'.*XXX.*\.png'
        regexp = QRegularExpression(pattern)
        png_valid.setRegularExpression(regexp)
        # png_valid.setLocale(QLocale('English'))
        self.pngname.setValidator(png_valid)
        self.pngname.setText(self._fname)
        self.pngname.textEdited.connect(self._setcheck)

        self.setWindowTitle("Insert PNG files template name")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.hlay = QHBoxLayout()
        self.hlay.addWidget(nameLabel)
        self.hlay.addWidget(self.pngname)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.okbtn = self.buttonBox.button(QDialogButtonBox.Ok)

        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._getpng)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay = QVBoxLayout()
        self.vlay.addLayout(self.hlay)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def _getpng(self):
        self._fname = self.pngname.text()
        self._okexit = True

    def _setcheck(self):
        if self.pngname.hasAcceptableInput():
            self.okbtn.setEnabled(True)
        else:
            self.okbtn.setEnabled(False)
            #self.buttonBox.

class TCDDialog(QDialog):
    def __init__(self, parent=None, maxval=1, texts: tp.List[str] = ["VTCD", "NM"]):
        super().__init__(parent)

        self.setWindowTitle(f"{texts[0]} Loader")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.vlay = QVBoxLayout()
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        self._maxval = maxval
        self._fname = ""
        self._val = None
        self._cube = None
        self._cubefname = "No file"

        message = QLabel(f"{texts[1]} number:")
        self.nmline = QLineEdit()
        nmval = QIntValidator(1, self._maxval)
        nmval.setLocale(QLocale('English'))
        self.nmline.setValidator(nmval)
        self.nmline.setText("1")
        self.nmline.editingFinished.connect(self._setval)

        openbutton = QPushButton('OpenCube', self)
        openbutton.clicked.connect(self.open)
 
        self.editline = QLineEdit()
        self.editline.setText("{}".format(self._cubefname))
        self.editline.setEnabled(False)
        # grid.addWidget(message)
        grid.addWidget(message, 0, 0)
        grid.addWidget(self.nmline, 0, 1)
        grid.addWidget(openbutton, 0, 2)
        grid.addWidget(self.editline, 0, 3)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setval)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def open(self):
        self._cube = QFileDialog.getOpenFileName(self, 'Select cube file', '.','*.cube')[0]
        self._cubefname = os.path.basename(self._cube)
        self.editline.setText("{}".format(self._cubefname))
 
    def _setval(self):
        self._vib = int(self.nmline.text())


class VTCDDialog(QDialog):
    def __init__(self, parent=None, nvib=1):
        super().__init__(parent)

        self.setWindowTitle("VTCD Loader")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.vlay = QVBoxLayout()
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        self._nvib = nvib
        self._fname = ""
        self._vib = None
        self._cube = None
        self._cubefname = "No file"

        message = QLabel("NM number:")
        self.nmline = QLineEdit()
        nmval = QIntValidator(1, self._nvib)
        nmval.setLocale(QLocale('English'))
        self.nmline.setValidator(nmval)
        self.nmline.setText("1")
        self.nmline.editingFinished.connect(self._setvib)

        openbutton = QPushButton('OpenCube', self)
        openbutton.clicked.connect(self.open)
 
        self.editline = QLineEdit()
        self.editline.setText("{}".format(self._cubefname))
        self.editline.setEnabled(False)
        # grid.addWidget(message)
        grid.addWidget(message, 0, 0)
        grid.addWidget(self.nmline, 0, 1)
        grid.addWidget(openbutton, 0, 2)
        grid.addWidget(self.editline, 0, 3)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setvib)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def open(self):
        self._cube = QFileDialog.getOpenFileName(self, 'Select cube file', '.','*.cube')[0]
        self._cubefname = os.path.basename(self._cube)
        self.editline.setText("{}".format(self._cubefname))
 
    def _setvib(self):
        self._vib = int(self.nmline.text())



class ETCDDialog(QDialog):
    def __init__(self, parent=None, nstate=1):
        super().__init__(parent)

        self.setWindowTitle("ETCD Loader")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.vlay = QVBoxLayout()
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        self._nstate = nstate
        self._fname = ""
        self._state = None
        self._cube = None
        self._cubefname = "No file"

        message = QLabel("Transition number:")
        self.stateline = QLineEdit()
        stval = QIntValidator(1, self._nstate)
        stval.setLocale(QLocale('English'))
        self.stateline.setValidator(stval)
        self.stateline.setText("1")
        self.stateline.editingFinished.connect(self._setstate)

        openbutton = QPushButton('OpenCube', self)
        openbutton.clicked.connect(self.open)
 
        self.editline = QLineEdit()
        self.editline.setText("{}".format(self._cubefname))
        self.editline.setEnabled(False)
        # grid.addWidget(message)
        grid.addWidget(message, 0, 0)
        grid.addWidget(self.stateline, 0, 1)
        grid.addWidget(openbutton, 0, 2)
        grid.addWidget(self.editline, 0, 3)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setstate)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def open(self):
        self._cube = QFileDialog.getOpenFileName(self, 'Select cube file', '.','*.cube')[0]
        self._cubefname = os.path.basename(self._cube)
        self.editline.setText("{}".format(self._cubefname))
 
    def _setstate(self):
        self._state = int(self.stateline.text())

class EditDoubleLine():
    """
    Edit a double value in a dialog
    return a QHBox layout with a QLabel and a QLineEdit
    """
    def __init__(self, text: str, default: float, validator: QDoubleValidator) -> None:
        self._hlay = QHBoxLayout()
        message = QLabel(text)
        self._hlay.addWidget(message)
        self._line = QLineEdit()
        self._line.setValidator(validator)
        self._line.setText(f"{default:12.6E}")
        self._hlay.addWidget(self._line)
        self._line.editingFinished.connect(self._setedit)
        self._edit = False
        self._default = default

    def _getvalue(self) -> float:
        return float(self._line.text())

    def _setedit(self):
        if not fuzzy_equal(self._getvalue(), self._default, tol=1E-10):
            self._edit = True

    @property
    def edit(self):
        return self._edit


class EditIntLine():
    """
    Edit a integer value in a dialog
    return a QHBox layout with a QLabel and a QLineEdit
    """
    def __init__(self, text: str, default: int, validator: QIntValidator) -> None:
        self._hlay = QHBoxLayout()
        message = QLabel(text)
        self._hlay.addWidget(message)
        self._line = QLineEdit()
        self._line.editingFinished.connect(self._setedit)
        self._line.setValidator(validator)
        self._line.setText(f"{default}")
        self._hlay.addWidget(self._line)
        self._edit = False
        self._default = default

    def _getvalue(self) -> int:
        return int(self._line.text())

    def _setedit(self):
        if self._getvalue() != self._default:
            self._edit = True

    @property
    def edit(self):
        return self._edit


class StreamLineSetupDialog(QDialog):
    """Dialog for setting up stream lines

    Args:
        QDialog (_type_): _description_
    """

    def __init__(self,
                 vfmax: float,
                 vfmin: float,
                 mspeed: float,
                 maxval: float,
                 nseeds: int,
                 scale: float,
                 direction: bool,
                 showellipse: bool,
                 showbar: bool = False,
                 parent: tp.Optional[tp.Union[QDialog, None]] = None,
                 ) -> None:
        """ initialize the dialog. Requires a dictionary with the parameters,
            the maximum norm value in the field and optionally a parent dialog

        Args:
            params (tp.Dict[str, tp.Any]): dictionary with the parameters
            maxval (float): maximum norm value in the field
            parent (tp.Optional[tp.Union[QDialog, None]]): Not required
        """
        super().__init__(parent)
        self._vfmax = vfmax
        self._vfmin = vfmin
        self._mspeed = mspeed
        self._nseeds = nseeds
        self._scale = scale
        self._direction = direction
        self._showellipse = showellipse
        self._showbar = showbar
        self._recalseeds = False
        self._redrawstream = False
        # print(f"vfmax:{self._vfmax} vfmin:{self._vfmin} mspeed:{self._mspeed} nseeds:{self._nseeds} scale:{self._scale}")
        self.setWindowTitle("Stream Line Setup Dialog")
        self.vlay = QVBoxLayout()
        message = QLabel(f"Max norm value:{maxval:12.6E}")
        self.vlay.addWidget(message)
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        message = QLabel("""Bound values as fraction
 of the max val:""")
        grid.addWidget(message,0, 0)
        message = QLabel("Upper denominator:")
        self.isoline = QLineEdit()
        maxvalid = QDoubleValidator()
        # validator for upper bound
        maxvalid.setRange(1, 1E12) # FIXME this shit
        maxvalid.setLocale(QLocale('English'))
        self._upmes = EditDoubleLine("Upper Bound", vfmax, maxvalid)
        grid.addLayout(self._upmes._hlay, 1, 0)
        minvalid = QDoubleValidator()
        minvalid.setLocale(QLocale('English'))
        minvalid.setRange(self._upmes._getvalue(),1E12)
        self._domes = EditDoubleLine("Lower bound", vfmin, minvalid)
        grid.addLayout(self._domes._hlay, 2, 0)
        self._shdir = QCheckBox("Show direction")
        self._shdir.setChecked(direction)
        self._shdir.stateChanged.connect(self._setdir)
        self._shell = QCheckBox("Show ellipsoids")
        self._shell.setChecked(showellipse)
        self._shell.stateChanged.connect(self._setell)
        self._shbar = QCheckBox("Show ColorBar")
        self._shbar.setChecked(showbar)
        self._shbar.stateChanged.connect(self._setbar)
        grid.addWidget(self._shdir, 3, 0)
        grid.addWidget(self._shell, 4, 0)
        grid.addWidget(self._shbar, 5, 0)

        # Second column
        message = QLabel("""Minimum speed for streamlines integration""")
        grid.addWidget(message,0 , 1)
        spdvalid = QDoubleValidator()
        spdvalid.setLocale(QLocale('English'))
        spdvalid.setRange(1E-12, maxval)
        self._spmes = EditDoubleLine("Minimum speed", mspeed, spdvalid)
        grid.addItem(self._spmes._hlay, 1, 1)
        seedvalid = QIntValidator()
        seedvalid.setLocale(QLocale('English'))
        # BUG limit hardcoded
        seedvalid.setRange(1, 500)
        self._seedline = EditIntLine("Number of seeds", nseeds, seedvalid)
        grid.addItem(self._seedline._hlay, 2, 1)
        scaleval = QDoubleValidator()
        scaleval.setLocale(QLocale('English'))
        scaleval.setRange(.2, 10.)
        self._scalemol = EditDoubleLine("Ellipsoid scaling factor", scale, scaleval)
        grid.addItem(self._scalemol._hlay, 3, 1)
        self._genseeds = QPushButton('Resample the ellissoide', self)
        self._genseeds.clicked.connect(self._setresample)
        hlay_tmp = QHBoxLayout()
        hlay_tmp.addWidget(self._genseeds)
        grid.addItem(hlay_tmp, 4, 1)

        # add text to the dialog
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setvals)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def _setvals(self):
        self._vfmax = self._upmes._getvalue()
        if self._upmes.edit:
            self._redrawstream = True
        self._vfmin = self._domes._getvalue()
        if self._domes.edit:
            self._redrawstream = True
        self._mspeed = self._spmes._getvalue()
        if self._spmes.edit:
            self._redrawstream = True
        self._nseeds = self._seedline._getvalue()
        if self._seedline.edit:
            self._recalseeds = True
            self._redrawstream = True
        self._scale = self._scalemol._getvalue()
        if self._scalemol.edit:
            self._recalseeds = True
            self._redrawstream = True
        # print(f"vfmax:{self._vfmax} vfmin:{self._vfmin} mspeed:{self._mspeed} nseeds:{self._nseeds} scale:{self._scale}")

    def _setresample(self):
        self._recalseeds = True
        self._redrawstream = True
        self._nseeds = self._seedline._getvalue()
        self._scale = self._scalemol._getvalue()
        # print(f"vfmax:{self._vfmax} vfmin:{self._vfmin} mspeed:{self._mspeed} nseeds:{self._nseeds} scale:{self._scale}")

    def _setdir(self):
        self._direction = self._shdir.isChecked()

    def _setell(self):
        self._showellipse = self._shell.isChecked()

    def _setbar(self):
        self._showbar = self._shbar.isChecked()

