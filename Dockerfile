FROM elynn/keras-tensorflow

RUN pip install --upgrade keras
RUN pip install pillow

COPY keras.json /root/.keras/

WORKDIR /src/ML-LayoutX/