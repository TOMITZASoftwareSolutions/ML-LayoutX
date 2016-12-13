FROM elynn/keras-tensorflow

RUN pip install --upgrade keras
RUN pip install pillow

COPY metrics.py /lib64/python2.7/site-packages/keras
COPY keras.json /root/.keras/

WORKDIR /src/ML-LayoutX/