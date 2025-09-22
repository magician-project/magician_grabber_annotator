FROM python:3.12

RUN apt-get update
RUN apt-get install -y build-essential libgtk-3-dev libnotify-bin
RUN pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/debian-9 wxPython

RUN pip install opencv-python numpy

WORKDIR /app

CMD [ "python3", "wxAnnotator.py" ]