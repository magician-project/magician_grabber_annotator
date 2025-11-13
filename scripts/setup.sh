#!/bin/bash



DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Welcome to the FORTH Magician repo automatic setup, everything will be installed @ `pwd`"
sleep 1


#Simple dependency checker that will apt-get stuff if something is missing
SYSTEM_DEPENDENCIES="python3-pip python3-venv build-essential cmake unzip wget git"

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
#------------------------------------------------------------------------------




cd "$DIR"
if [ -f magician_vision_classifier/README.md ]
then
echo "Vision classifier already downloaded skipping.."
else
  echo "         Do you want to download the Magician Vision Classifier ? " 
  echo "                    (You probably need this )" 
  echo
  echo -n " (Y/N)?"
  read answer
  #answer="Y"
  if test "$answer" != "N" -a "$answer" != "n";
  then
     git clone git@github.com:magician-project/magician_vision_classifier.git
     cd magician_vision_classifier

     items="allclass_convnext_tiny allclass_verysmall_cnn allclass_efficientnet_v2_s allclass_resnet18 allclass_resnext50 binary_small_cnn binary_resnet18"
     files=""
     for item in $items; do
         files="$files ${item}.pth ${item}.json"
     done

     for item in $files; do
         if [ -f "$item" ]; then
           echo "Skipping $item (already exists)"
         else
           wget -O "$item" "http://ammar.gr/magician/ckpts/$item"
        fi
     done

     python3 -m venv venv
     source venv/bin/activate
     python3 -m pip install -r requriements.txt 
     python3 -m pip install wxPython
     deactivate

     cd "$DIR"
  fi
fi



cd "$DIR"
if [ -f magician_grabber_annotator/README.md ]
then
echo "Vision annotator already downloaded skipping.."
else
  echo "         Do you want to download the Magician Vision Annotator ? " 
  echo "                    (You probably need this )" 
  echo
  echo -n " (Y/N)?"
  read answer
  #answer="Y"
  if test "$answer" != "N" -a "$answer" != "n";
  then 
     git clone git@github.com:magician-project/magician_grabber_annotator.git
     cd magician_grabber_annotator

     cd "$DIR"
  fi
fi


cd "$DIR"
if [ -f magician_general_visual_perception/README.md ]
then
echo "Visual Perception module already downloaded skipping.."
else
  echo "         Do you want to download the Magician Visual Perception ? " 
  echo "                    (You probably need this )" 
  echo
  echo -n " (Y/N)?"
  read answer
  #answer="Y"
  if test "$answer" != "N" -a "$answer" != "n";
  then 
     git clone git@github.com:magician-project/magician_general_visual_perception.git
     cd magician_general_visual_perception

     scripts/setup.sh
     scripts/downloadPretrained.sh
     cd "$DIR"
  fi
fi






cd "$DIR"
if [ -f magician_grabber/README.md ]
then
echo "Magician Grabber already downloaded skipping.."
else
  echo "         Do you want to download the Magician Grabber annotator ? " 
  echo "    (You probably don't need this except if you have a Magician Sensor )" 
  echo
  echo -n " (Y/N)?"
  read answer
  #answer="Y"
  if test "$answer" != "N" -a "$answer" != "n";
  then
    git clone git@github.com:magician-project/magician_grabber.git
    cd magician_grabber

    scripts/build.sh
    make
  fi
fi






cd "$DIR"
if [ -f magician_main_board/README.md ]
then
echo "Magician Sensor Blueprints already downloaded skipping.."
else
  echo "         Do you want to download the Magician PCB Blueprints ? " 
  echo "                (You probably don't need this )" 
  echo
  echo -n " (Y/N)?"
  read answer
  #answer="Y"
  if test "$answer" != "N" -a "$answer" != "n";
  then
     git clone git@github.com:magician-project/magician_main_board
  fi
fi






exit 0
