#! /bin/bash
# Author: You Cunhan
# youch@shanghaitech.edu.cn

# To install gc, g++ and other required package
echo "-------Start Step1 Download dependencies-------"
sudo apt update
sudo apt install build-essential libncurses5 libncurses5-dev libx11-dev xorg-dev qemu git -y
echo "-------Finish Step1 Download dependencies-------"



# To download Pintos project and patched bochs/qemu
echo "-------Start Step2 Download and extract pintos and bochs/qemu-------"
git clone `YOUR_PINTOS_MIRROR` pintos
wget https://nchc.dl.sourceforge.net/project/bochs/bochs/2.6.2/bochs-2.6.2.tar.gz


# To extract *.tar.gz files

cd pintos
git remote remove origin
cd ..

echo "-------Finish Step2 Download and extract pintos and bochs-------"


echo "-------Start Step3 Configure bochs and gdb for pintos-------"

# Install bochs-2.6.2 for pintos with pintos script
PINTOSDIR=`pwd -P`/pintos
SRCDIR=`pwd -P`
DSTDIR=/usr/local
sudo env SRCDIR=$SRCDIR PINTOSDIR=$PINTOSDIR DSTDIR=$DSTDIR sh $PINTOSDIR/src/misc/bochs-2.6.2-build.sh

# Add PATH environment variable for Pintos

echo "" >> $HOME/.bashrc
echo "export PINTOSDIR=\"$PINTOSDIR\"" >> $HOME/.bashrc
echo "export PATH=\"$PINTOSDIR/src/utils:\$PATH\"" >> $HOME/.bashrc

# Patch for pintos-gdb
sed -i "/GDBMACROS=\/usr\/class\/cs140\/pintos\/pintos\/src\/misc\/gdb-macros/c\GDBMACROS=\$PINTOSDIR/src/misc/gdb-macros" $PINTOSDIR/src/utils/pintos-gdb

# Compile the remaining Pintos utilities
version=$(lsb_release -r --short)
if [ ${version:0:2} = "20" ]; then
    sed -i '10,10d' $PINTOSDIR/src/utils/squish-pty.c
    sed -i '287,292d' $PINTOSDIR/src/utils/squish-pty.c
    sed -i '11,11d' $PINTOSDIR/src/utils/squish-unix.c
fi
cd $PINTOSDIR/src/utils
make
cd $SRCDIR
## Idle
sudo rm ./bochs-2.6.2.tar.gz
chmod -R 777 $PINTOSDIR
echo "-------Finish Step3 Configure bochs and gdb for pintos-------"

echo "-------Finish installing-------"
echo "You can type \"cd $PINTOSDIR/src/threads && make check\" to test whether everything is ok (7 of 27 pass), it may take a while."
