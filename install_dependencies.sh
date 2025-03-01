#!/bin/bash
# Script to install QGIS and GDAL dependencies for health indices processor

set -e  # Exit on error

echo "Installing QGIS and GDAL dependencies..."

# Update package lists
apt-get update

# Install GDAL and Python bindings
echo "Installing GDAL and Python bindings..."
apt-get install -y python3-gdal gdal-bin

# Install QGIS dependencies
echo "Installing QGIS dependencies..."
apt-get install -y python3-pyqt5 python3-pyqt5.qsci python3-pyqt5.qtwebkit

# Add QGIS repository
echo "Adding QGIS repository..."
apt-get install -y gnupg software-properties-common
wget -qO - https://qgis.org/downloads/qgis-2022.gpg.key | gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/qgis-archive.gpg --import
chmod a+r /etc/apt/trusted.gpg.d/qgis-archive.gpg
add-apt-repository "deb https://qgis.org/ubuntu $(lsb_release -c -s) main"

# Update package lists again
apt-get update

# Install QGIS and Python bindings
echo "Installing QGIS and Python bindings..."
apt-get install -y qgis python3-qgis qgis-plugin-grass

# Verify installations
echo "Verifying installations..."
python3 -c "from osgeo import gdal; print('GDAL version:', gdal.__version__)"
python3 -c "import qgis.core; print('QGIS available')"

echo "Installation complete!"
echo "You may need to restart your system or Python environment for changes to take effect." 