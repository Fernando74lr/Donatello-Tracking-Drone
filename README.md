# Vision Tracking Drone Guided by Deep Convolutional Neural Network for Recording Third-Person Cycling Scenes
En los últimos años los drones han tomado gran relevancia en tecnologías de vanguardia, por lo que hoy en día los drones más modernos pueden ser controlados a grandes distancias y cuentan con diferentes elementos como como cámaras, sensores, GPS, entre otros, haciéndolos perfectos para uso en múltiples entornos de trabajo, proyectos, oportunidades y más. Los drones de rastreo son capaces de apuntar a objetos específicos y pueden proporcionar resultados increíbles para la fotografía y grabación de vídeo, pudiendo capturar planos, puntos de vista y ángulos que por otros medios convencionales serían muy difíciles. El objetivo de este proyecto es desarrollar un dron de seguimiento de ciclistas para grabar escenas en perspectiva de tercera persona. Los resultados obtenidos en este proyecto muestran un vuelo completamente autónomo mediante un triple controlador PID trabajando en paralelo, lo que se consiguió gracias al algoritmo de visión mediante red neuronal convolucional profunda convolucional que fue bastante preciso y óptimo, obteniendo al final de la rutina la grabación correspondiente, además cuenta con una interfaz amigable e intuitiva para el usuario final

[YOLO](https://pjreddie.com/darknet/yolo/) es un modelo capaz de detectar 80 diferentes objetos que fueron previamente entrenados, y dentro de ellos se encuentran nuestros dos objetivos, una persona y una bicicleta. Este algoritmo se llama as ́ı (You Only Look Once) ya que solo necesita de un solo frame para detectar cualquier objeto que esté presente dentro de las clases permitidas. Además de ser muy rápido y ser beneficiado si se corre en una GPU NVIDIA utilizando [CUDA](https://developer.nvidia.com/cuda-toolkit) la cual es una plataforma de computación paralela que permite al software en el que se encuentre instalado el uso de la tarjeta de procesamiento gráfica.

Los pasos para poder correr el siguiente programa a través de un dron Tello (Para el desarrollo de este prototipo fue utilizada una Laptop con sistema operativo Ubuntu 22.04 y una tarjeta gráfica NVIDIA GeForce RTX 2070) son:

# Crear ambiente
Para tener en orden nuestras paqueterías de python primero vamos a crear un ambiente llamado "donatello" el cual tiene la versión 3.6 de python
``` 
conda create -n donatello python=3.6
```

Activamos el ambiente donatello para asegurarnos que estemos en el ambiente correcto al momento de hacer la instalación de todas las paqueterías necesarias
```
source activate donatello
```

# Instalación de las paqueterías
Estando dentro de nuestro ambiente vamos a instalar todas las paqueterías necesarias para correr nuestro vision tracking drone, la lista de los paquetes y versiones a instalar están dentro del archivo requirements.txt por lo cual instalaremos haciendo referencia a ese archivo
```
pip install -r requirements.txt
```

# Descargar los pesos del modelo entrenado 
Para poder correr el modelo de YOLO tendremos que descargar los pesos de la red neuronal, los pesos son los valores que tienen todas las conexiones entre las neuronas de la red neuronal de YOLO, este tipo de modelos son computacionalmente muy pesados de entrenar desde cero por lo cual descargar el modelo pre entrenado es una buena opción.

```
bash video_stream/weights/download_weights.sh
```

Movemos los pesos descargados a la carpeta llamada weights
```
mv yolov3.weights video_stream/weights/
```

# Conexión del dron
Conectar la computadora a la misma red del dron Tello

# Correr la interfaz
Utilizar el siguiente comando para correr la interfaz:
```
python manage.py runserver
```
Entrar en su navegador de preferencia al link que arroja en la terminal, usualmente: 127.0.0.1:8000

# Videos
[Explicación](https://youtu.be/WRJ4od2K-6o)

[Demostración](https://youtu.be/dUZFRqen_c8)


¡GRACIAS por su interés y atención!

![interface-dron](https://user-images.githubusercontent.com/43561384/205850407-3d99bb4c-02f7-4a63-84f5-4e2840622dc4.png)


# Referencias
Este proyecto toma como base el siguiente [PyTorch YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
