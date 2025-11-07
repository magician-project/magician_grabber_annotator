#!/bin/bash



DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Will populate Annotator+Classifier @ `pwd`"

git clone git@github.com:magician-project/magician_vision_classifier.git
git clone git@github.com:magician-project/magician_grabber_annotator.git
git clone git@github.com:magician-project/magician_grabber.git
git clone git@github.com:magician-project/magician_main_board


exit 0
