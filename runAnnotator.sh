#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"



if [ -d ../../../venv/ ]
then
echo "Found a central virtual environment, activating it with priority!" 
source ../../../venv/bin/activate
elif [ -d venv/ ]
then
echo "Found a virtual environment" 
source venv/bin/activate
else
echo "Did not find an existing venv.."

SYSTEM_DEPENDENCIES="python3-venv python3-pip zip wget curl"

for REQUIRED_PKG in $SYSTEM_DEPENDENCIES
do
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo "Checking for $REQUIRED_PKG: $PKG_OK"
if [ "" = "$PKG_OK" ]; then

  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."

  #If this is uncommented then only packages that are missing will get prompted..
  #sudo apt-get --yes install $REQUIRED_PKG

  #if this is uncommented then if one package is missing then all missing packages are immediately installed..
  sudo apt-get install $SYSTEM_DEPENDENCIES  
  break
fi
done

python3 -m venv venv
source venv/bin/activate
python3 -m pip install wxPython opencv-python numpy

fi




python3 wxAnnotator.py $@

exit 0
