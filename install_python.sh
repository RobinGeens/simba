#!/bin/bash
PYVER=3.11.9
INSTALL_DIR=/volume1/users/rgeens/simba/Python-$PYVER 

wget https://www.python.org/ftp/python/$PYVER/Python-$PYVER.tgz
tar -xzf Python-$PYVER.tgz
cd Python-$PYVER

# Configure for custom install with shared libs and headers
./configure --prefix=$INSTALL_DIR --enable-optimizations --enable-shared

# Build and install
make -j$(nproc)
make install

# Update environment variables to use custom Python
export PATH=$INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH

# Check that Python.h is now present
find $INSTALL_DIR/include -name Python.h
