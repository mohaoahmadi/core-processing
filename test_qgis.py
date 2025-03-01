#!/usr/bin/env python3
from qgis.core import (
        QgsApplication,
        QgsRasterLayer,
        QgsRasterCalculator,
        QgsRasterCalculatorEntry,
        QgsCoordinateReferenceSystem,
        QgsProject
    )

# Initialize QGIS Application
QgsApplication.setPrefixPath("/usr", True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Print QGIS version
print("QGIS version:", Qgis.QGIS_VERSION)

# Exit
qgs.exitQgis()
