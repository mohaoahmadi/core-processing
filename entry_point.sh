#!/bin/bash

source venv/bin/activate

export QGIS_PREFIX_PATH="/usr/share/qgis" && \
export PYTHONPATH="/usr/share/qgis/python:/usr/lib/python3/dist-packages:$PYTHONPATH" && \
export LD_LIBRARY_PATH="/usr/lib/qgis:$LD_LIBRARY_PATH" && \
export QT_QPA_PLATFORM="offscreen"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
