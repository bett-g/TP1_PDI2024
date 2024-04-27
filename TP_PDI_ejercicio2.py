import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_lineas_horizontales(image, umbral):
    suma_filas = np.sum(image, axis=1)
    lineas_horizontales = []

    for idx, suma in enumerate(suma_filas):
        if suma > umbral:
            lineas_horizontales.append(idx)

    return lineas_horizontales

def detectar_lineas_verticales(image, umbral):
    suma_columnas = np.sum(image, axis=0)
    lineas_verticales = []

    for idx, suma in enumerate(suma_columnas):
        if suma > umbral:
            lineas_verticales.append(idx)

    return lineas_verticales

def detectar_celdas(image, lineas_horizontales, lineas_verticales):
    celdas = []

    for i in range(len(lineas_horizontales) - 1):
        for j in range(len(lineas_verticales) - 1):
            x1 = lineas_verticales[j]
            y1 = lineas_horizontales[i]
            x2 = lineas_verticales[j + 1]
            y2 = lineas_horizontales[i + 1]
            celdas.append((x1, y1, x2, y2))

    return celdas

def opcion_marcada(cell, umbral):
    mean_intensity = np.mean(cell)
    return mean_intensity < umbral

def obtener_respuestas_marcadas(image, umbral):
    """
    Detecta las celdas de respuesta en una imagen de examen de opción múltiple y visualiza los resultados.
    Args:
        image (numpy.ndarray): Imagen en escala de grises del examen.
        umbral (int): Umbral para la binarización de la imagen.
    Returns:
        dict: Diccionario que asocia el número de pregunta con la posición de la celda marcada: {numero_pregunta: (x0, y0, x1, y1), ...}.
    """

    # Preprocesamiento de la imagen
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Desvanecer el ruido
    _, thresh = cv2.threshold(image, umbral, 255, cv2.THRESH_BINARY)  # Binarización

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Diccionario para almacenar las respuestas del estudiante
    respuestas_estudiante = {}

    # Filtrar contornos por área (celdas de respuesta)
    for idx, contour in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Considerar solo contornos con un área mínima
            respuestas_estudiante[idx] = (x, y, x + w, y + h)  # Almacenar posición de la celda marcada

    # Mostrar la imagen con los rectángulos
    image_visualize = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in respuestas_estudiante.values():
        cv2.rectangle(image_visualize, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Dibujar rectángulo
    plt.figure(figsize=(8, 6))
    plt.imshow(image_visualize)
    plt.title("Respuestas marcadas detectadas")
    plt.axis("off")
    plt.show()

    return respuestas_estudiante

################COMPARAR CON RESPUESTAS CORRECTAS
"""def comparar_rtas_correctas(imagen_umbralizada):
    
    result_image = np.empty(imagen_umbralizada.shape)

    rtas_correctas = 'AABADBDCBADACCDDACCDBACCC'

    # Convertir el texto en una cadena de texto
    text = pytesseract.image_to_string(imagen_umbralizada, config='--oem 3 --psm 6')

    lineas = text.split('\n')

    for i, line in enumerate(lineas):
        if line.startswith("Pregunta"):
            question_number = int(line.split()[1].strip(':'))
            answer = line.split()[-1]
            if answer == rtas_correctas[question_number - 1]:
                print(f"Pregunta {question_number}: OK")
            else:
                print(f"Pregunta {question_number}: MAL")
"""
# Cargar la imagen y procesarla
imagen = cv2.imread('C:\\Users\\betsa\\OneDrive\\Escritorio\\TUIA2024\\PDI\\TP\\TP1_PDI2024\\img\\multiple_choice_1.png', cv2.IMREAD_GRAYSCALE)
_, imagen_umbralizada = cv2.threshold(imagen, 150, 255, cv2.THRESH_BINARY_INV)

imagen_umbralizada = imagen_umbralizada[158:969, 248:539]

imagen_vacia = cv2.imread('C:\\Users\\betsa\\OneDrive\\Escritorio\\TUIA2024\\PDI\\TP\\TP1_PDI2024\\img\\multiple_choice_5.png', cv2.IMREAD_GRAYSCALE)
imagen_vacia = imagen_vacia[199:988,495:741]
plt.imshow(imagen_vacia, cmap="gray")
img_matriz = np.array(imagen_vacia)

print(img_matriz)
# Parámetros de umbralización específicos para detectar líneas
umbral_lineas_horizontales = 250 * imagen_umbralizada.shape[1]
umbral_lineas_verticales = 100 * imagen_umbralizada.shape[0]

# Detectar líneas horizontales y verticales
lineas_horizontales = detectar_lineas_horizontales(imagen_umbralizada, umbral_lineas_horizontales)
lineas_verticales = detectar_lineas_verticales(imagen_umbralizada, umbral_lineas_verticales)

# Detectar celdas
celdas = detectar_celdas(imagen_umbralizada, lineas_horizontales, lineas_verticales)

# Respuestas correctas
respuestas_correctas = {
    1: 'A', 2: 'A', 3: 'B', 4: 'A', 5: 'D', 6: 'B', 7: 'B', 8: 'C', 9: 'B', 10: 'A',
    11: 'D', 12: 'A', 13: 'C', 14: 'C', 15: 'D', 16: 'B', 17: 'A', 18: 'C', 19: 'C', 20: 'D',
    21: 'B', 22: 'A', 23: 'C', 24: 'C', 25: 'C'
}

# Umbral para determinar si una opción está marcada
umbral_opcion_marcada = 200

# Obtener respuestas del estudiante
respuestas_estudiante = obtener_respuestas_marcadas(imagen_umbralizada, umbral_opcion_marcada)

#comparar_rtas_correctas(imagen_umbralizada)