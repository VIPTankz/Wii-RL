#!/bin/bash

DOLPHINMPN_URL="https://github.com/Felk/dolphin.git"
DOLPHIN_DIR="dolphin"

if [ -d "$DOLPHIN_DIR" ]; then
    echo "Dolphin MPN already cloned."
else
    echo "Cloning Dolphin MPN..."
    git clone --depth=1 $DOLPHINMPN_URL $DOLPHIN_DIR
    cd $DOLPHIN_DIR
    git submodule update --init --recursive --progress --depth=1
    echo "Dolphin cloned successfully."
    cd ..
fi

# Entering the source code
cd $DOLPHIN_DIR

# Compiling dolphin
mkdir -p Build
cd Build
# Option "CMAKE_POLICY_VERSION_MINIMUM=3.5" is used for compability with newer CMakes
# Option "USE_SYSTEM_FMT=OFF" is used for compability for Arch Linux
cmake .. -DLINUX_LOCAL_DEV=true -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DUSE_SYSTEM_FMT=OFF
make -j $(nproc)
cp -r ../Data/Sys/ Binaries/

# Making dolphin portable
touch Binaries/portable.txt

# Deleting the old dolphin we had (if there is one)
rm -rf ../../dolphin0

# Copying the compiled dolphin
cp -r Binaries/ ../../dolphin0
cd ../..

# Deleting the source code
rm -rf $DOLPHIN_DIR
