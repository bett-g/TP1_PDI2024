import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_local_histogram_equalization(image, window_size):
    # Se obtienen las dimensiones de la imagen
    height, width = image.shape
    # Se calcula la mitad del tamaño de la ventana
    half_window = window_size // 2

    # Se agregan bordes para manejar los píxeles cercanos a los bordes de la imagen
    image_with_borders = cv2.copyMakeBorder(image,
                                            half_window,
                                            half_window,
                                            half_window,
                                            half_window,
                                            cv2.BORDER_REPLICATE)

    # Matriz vacía para guardar los resultados de la ecualización
    result_image = np.empty(image.shape)

    # Itera sobre los píxeles de la imagen original
    for i in range(half_window, height + half_window):
        for j in range(half_window, width + half_window):
            # Se extrae la ventana alrededor del píxel actual
            window = image_with_borders[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            # Aplica la ecualización de histograma a la ventana
            equalized_window = cv2.equalizeHist(window)
            # Guarda el píxel central de la ventana ecualizada
            result_image[i-half_window, j-half_window] = equalized_window[half_window, half_window]

    return result_image

# Carga la imagen original
ruta_imagen = 'img\Imagen_con_detalles_escondidos.tif'
imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Aplica la ecualización de histograma local con diferentes tamaños de ventana
img1 = apply_local_histogram_equalization(imagen_original, 3*3)
img2 = apply_local_histogram_equalization(imagen_original, 11*3)
img3 = apply_local_histogram_equalization(imagen_original, 33*3)

# Mostrar imágenes
ax1 = plt.subplot(221)
plt.title('Imagen Original')
plt.imshow(imagen_original, cmap='gray')

plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title('Ventana de 3x3')
plt.imshow(img1, cmap='gray')

plt.subplot(223, sharex=ax1, sharey=ax1)
plt.title('Ventana de 11x3')
plt.imshow(img2, cmap='gray')

plt.subplot(224, sharex=ax1, sharey=ax1)
plt.title('Ventana de 33x3')
plt.imshow(img3, cmap='gray')

plt.show()