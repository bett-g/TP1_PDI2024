import cv2 
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt 
import matplotlib.patches as mpatches

def cargar_img(rutas):
    return [cv2.imread(ruta, cv2.IMREAD_GRAYSCALE) for ruta in rutas]

def procesar_form(imagen_bool):
    img_rows = np.sum(imagen_bool, axis=1)
    filas_validas = img_rows > 500
    indices_validos = np.where(filas_validas)[0]
    umbral_espacio = 22

    primer_espacio_index = next((i for i in range(1, len(indices_validos)) if indices_validos[i] - indices_validos[i - 1] > umbral_espacio), None)
    if primer_espacio_index is None:
        return None, None, None, None, None

    indice_inicial = indices_validos[primer_espacio_index - 1] + 1
    indice_final = indices_validos[primer_espacio_index] - 1

    img_final = imagen_bool[indice_inicial:indice_final]

    img_cols = np.sum(img_final, axis=0)
    umbral_columna_valida = 20
    columnas_validas = img_cols > umbral_columna_valida

    inicio_sub_imgs = []
    fin_sub_imgs = []

    i = 0
    while i < len(columnas_validas):
        if not columnas_validas[i]:
            inicio = i
            while i < len(columnas_validas) and not columnas_validas[i]:
                i += 1
            fin = i - 1
            inicio_sub_imgs.append(inicio)
            fin_sub_imgs.append(fin)
        i += 1

    nombre = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[2]:fin_sub_imgs[2]]
    id = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[4]:fin_sub_imgs[4]]
    tipo = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[6]:fin_sub_imgs[6]]
    fecha = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[8]:fin_sub_imgs[8]]
    multiple_choice = imagen_bool[indice_final + 2:, :]

    return nombre, id, tipo, fecha, multiple_choice

def contar_componentes_conectados(img):
    f_point = img.astype(np.uint8)
    connectivity = 8
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(f_point, connectivity, cv2.CV_32S)
    caracteres = 0
    palabras = 0
    if num_labels <= 2:
        caracteres = num_labels - 1
        palabras = num_labels - 1
    else:   
        ind_ord = np.argsort(stats[:,0])
        stats_ord = stats[ind_ord]
        resultados = []
        for i in range(2, num_labels):
            fila_actual = stats_ord[i]
            fila_anterior = stats_ord[i - 1]
            suma = fila_actual[0] - (fila_anterior[0] + fila_anterior[2])        
            resultados.append(suma)    
        area_promedio = np.mean(stats_ord[:, 4])  # Calculamos el área promedio de los componentes
        umbral = 3 * area_promedio  # Usamos un múltiplo del área promedio como umbral
        espacios = sum(1 for valor in resultados if valor >= umbral)
        palabras = espacios + 1
        caracteres = num_labels + espacios - 1
    return caracteres, palabras

def validar_campos(examen):  
    nombre, legajo, tipo, dia, multiple_choice = procesar_form(examen)

    def validar_componente(componente, caracteres_requeridos, palabras_requeridas=None):
        caracteres, palabras = contar_componentes_conectados(componente)
        if palabras_requeridas is None:
            resultado = "OK" if caracteres == caracteres_requeridos else "MAL"
        else:
            resultado = "OK" if caracteres == caracteres_requeridos and palabras == palabras_requeridas else "MAL"
        return resultado

    diccionario_validacion = {
        "Nombre y Apellido": validar_componente(nombre, caracteres_requeridos=(1, 25), palabras_requeridas=2),
        "Legajo": validar_componente(legajo, caracteres_requeridos=8, palabras_requeridas=1),
        "Código": validar_componente(tipo, caracteres_requeridos=1),
        "Fecha": validar_componente(dia, caracteres_requeridos=8, palabras_requeridas=1)
    }

    return diccionario_validacion, multiple_choice, nombre

def detectar_circulos(imagen_binaria):
    circulos = cv2.HoughCircles(imagen_binaria, cv2.HOUGH_GRADIENT, dp=2, minDist=20, param1=50, param2=20, minRadius=10, maxRadius=20)
    circulos = sorted(circulos[0, :], key=lambda circle: circle[0])
    circulos = np.uint16(np.around(circulos))
    circulos = [np.append(circle, x + 1) for x, circle in enumerate(circulos)]
    circulos = np.array(circulos)
    return circulos

def procesar_renglones(multiple_choice_binario):
    img_rows = np.sum(multiple_choice_binario, 1)  # Calcula el número de píxeles blancos en cada fila
    umbral_columna_valida = 20  # Define el umbral mínimo de píxeles blancos por fila para considerarla válida
    columnas_validas = img_rows >= umbral_columna_valida  # Encuentra las filas válidas basadas en el umbral definido
    cambios = np.diff(columnas_validas)  # Encuentra los cambios entre filas válidas e inválidas
    indices_renglones = np.argwhere(cambios)  # Encuentra los índices de cambio entre filas válidas e inválidas
    indices_renglones = indices_renglones[1:]  # Descarta el primer cambio, ya que no indica un inicio de pregunta
    indices_renglones = indices_renglones.reshape((-1, 2))  # Agrupa los índices de cambio en pares representando rangos de filas
    return indices_renglones


def respuestas(multiple_choice_binario):
    respuestas = []  # Lista para almacenar las respuestas encontradas
    preguntas = []   # Lista para almacenar los números de pregunta correspondientes a las respuestas encontradas
    indices_renglones = procesar_renglones(multiple_choice_binario)  # Obtiene los rangos de filas válidas

    # Itera sobre los rangos de filas válidas donde podría haber preguntas
    for i in range(len(indices_renglones)):
        renglon = multiple_choice_binario[indices_renglones[i][0]:indices_renglones[i][1], 0:1000]  # Recorta la región de interés de la imagen
        imagen_binaria = renglon.astype(np.uint8) * 255  # Convierte la región recortada en una imagen binaria
        circulos = detectar_circulos(imagen_binaria)  # Detecta círculos en la imagen binaria
        tamaño_vecindad = 10  # Tamaño de la vecindad para comprobar si el círculo está relleno
        cuenta = 0  # Contador de círculos rellenos dentro de una pregunta

        # Itera sobre los círculos detectados en la imagen binaria
        for circulo in circulos:
            x, y = circulo[0], circulo[1]  # Coordenadas del centro del círculo
            vecindad_x = slice(max(0, x - tamaño_vecindad), min(imagen_binaria.shape[1], x + tamaño_vecindad + 1))  # Define la región en el eje X alrededor del centro del círculo
            vecindad_y = slice(max(0, y - tamaño_vecindad), min(imagen_binaria.shape[0], y + tamaño_vecindad + 1))  # Define la región en el eje Y alrededor del centro del círculo
            píxeles_vecindad = imagen_binaria[vecindad_y, vecindad_x]  # Extrae la vecindad de píxeles alrededor del centro del círculo
            está_relleno = np.count_nonzero(píxeles_vecindad == 255) >= píxeles_vecindad.size * 0.5  # Verifica si al menos el 50% de los píxeles en la vecindad están rellenos de blanco
            if está_relleno:  # Si el círculo está relleno de blanco
                preguntas.append(i+1)  # Guarda el número de pregunta
                respuestas.append(circulo[3])  # Guarda el identificador de respuesta
                cuenta += 1  # Incrementa el contador de círculos rellenos
        
        # Determina la respuesta para la pregunta actual
        if cuenta == 0 or cuenta > 1:  # Si no se detecta ningún círculo relleno o se detectan más de uno
            preguntas.append(i + 1)  # Guarda el número de pregunta como no respondida (0)
            respuestas.append(0)
        
    # Crea un diccionario que mapea los números de pregunta a los identificadores de respuesta y lo devuelve
    diccionario_respuestas = dict(zip(preguntas, respuestas))
    return diccionario_respuestas

def corregir_examen(respuestas):
    respuestas_correctas = {
        1: 1, 2: 1, 3: 2, 4: 1, 5: 4, 6: 2, 7: 2, 8: 3, 9: 2, 10: 1, 11: 4,
        12: 1, 13: 3, 14: 3, 15: 4, 16: 2, 17: 1, 18: 3, 19: 3, 20: 4, 21: 2,
        22: 1, 23: 3, 24: 3, 25: 3
    }

    correcciones = {}  
    contador_ok = 0  

    for clave in respuestas:
        if clave in respuestas_correctas:  
            if respuestas[clave] == respuestas_correctas[clave]:
                correcciones[clave] = "OK"
                contador_ok += 1  
            else:
                correcciones[clave] = "MAL"

    aprobado = contador_ok >= 20
    return correcciones, aprobado

def main():
    rutas_examenes = ['img/multiple_choice_1.png', 
                      'img/multiple_choice_2.png', 
                      'img/multiple_choice_3.png', 
                      'img/multiple_choice_4.png', 
                      'img/multiple_choice_5.png']

    examenes = cargar_img(rutas_examenes)
    resultados_concatenados = None  

    # Inicializar contadores
    contador_aprobados = 0
    contador_reprobados = 0

    for indice, examen in enumerate(examenes):
        img_th = examen < 200   
        dic_resultados_validacion, imagen_multi_choice, nombre = validar_campos(img_th)
        dic_respuestas_preguntas = respuestas(imagen_multi_choice)
        dic_examen_corregido, aprobado = corregir_examen(dic_respuestas_preguntas)

        if aprobado:
            nombre_resultado = 255 - nombre 
            contador_aprobados += 1  # Incrementar contador de aprobados
        else:
            nombre_resultado = 255 + nombre
            contador_reprobados += 1  # Incrementar contador de reprobados

        # Concatenación de resultados
        if resultados_concatenados is None:
            resultados_concatenados = nombre_resultado
        else:
            resultados_concatenados = np.vstack((resultados_concatenados, nombre_resultado))

        # Impresión de resultados
        print(f"EXAMEN {indice+1} RESULTADOS:")
        print("Validación Campos:")
        for campo, resultado in dic_resultados_validacion.items():
            print(f"{campo}: {resultado}")
        print("\nCorrección:")
        for pregunta, resultado in dic_examen_corregido.items():
            print(f"Pregunta {pregunta}: {resultado}")

    # Normalización de los valores para que puedan ser guardados en una imagen
    min_val = np.min(resultados_concatenados)
    max_val = np.max(resultados_concatenados)
    resultados_normalizados = 255 * (resultados_concatenados - min_val) / (max_val - min_val)
    resultados_normalizados = resultados_normalizados.astype(np.uint8)
    # Guardar la imagen
    cv2.imwrite('resultados_examen.png', resultados_normalizados)

    # Asegurar que la imagen no esté completamente en blanco
    print("Valor mínimo en la imagen:", np.min(resultados_normalizados))
    print("Valor máximo en la imagen:", np.max(resultados_normalizados))

    # Crear una copia de la imagen para modificar los colores
    resultados_colores = cv2.cvtColor(resultados_normalizados, cv2.COLOR_GRAY2BGR)

    # Definir colores en formato BGR
    color_aprobado = (0, 255, 0)  # Verde
    color_reprobado = (0, 0, 255)  # Rojo
    mascara_blancos = np.all(resultados_colores == [255, 255, 255], axis=2)
    resultados_colores[mascara_blancos] = [255, 255, 255]

    # Obtener los índices de los nombres de los alumnos aprobados y reprobados
    aprobados_indices = np.where(resultados_normalizados == 0)  # En la imagen en blanco y negro, los aprobados están en 0 (negro)
    reprobados_indices = np.where(resultados_normalizados == 255)  # En la imagen en blanco y negro, los reprobados están en 255 (blanco)

    # Pintar los nombres de los alumnos aprobados en verde y los reprobados en rojo
    resultados_colores[aprobados_indices] = color_aprobado
    resultados_colores[reprobados_indices] = color_reprobado

    # Mostrar la imagen con los nombres de los alumnos coloreados y más grande
    fondo_blanco = np.full_like(resultados_colores, (255, 255, 255), dtype=np.uint8)

    # Superponer la imagen coloreada sobre el fondo blanco
    imagen_con_fondo_blanco = np.where(resultados_colores == 0, resultados_colores, fondo_blanco)
    plt.figure(figsize=(7, 4))  # tamaño de la figura

    # Mostrar la imagen con los nombres de los alumnos coloreados
    plt.imshow(cv2.cvtColor(imagen_con_fondo_blanco, cv2.COLOR_BGR2RGB))
    plt.title('RESULTADOS DEL EXAMEN')
    plt.xlabel('ALUMNOS APROBADOS (en verde)  /  ALUMNOS REPROBADOS (en rojo)')
    plt.ylabel('NOMBRES DE LOS ALUMNOS')
    plt.xticks([])
    plt.yticks([])

    # Agregar leyenda
    aprobado_patch = mpatches.Patch(color='green', label='Alumnos Aprobados')
    reprobado_patch = mpatches.Patch(color='red', label='Alumnos Reprobados')
    plt.legend(handles=[aprobado_patch, reprobado_patch], loc='upper right')
    plt.show()

    print("RESULTADOS GUARDADOS: resultados_examen.png")
    print("Los nombres de los alumnos que han APROBADO se muestran en color VERDE en la imagen.")
    print("Los nombres de los alumnos que han REPROBADO se muestran en color ROJO en la imagen.")
    # Imprimir cantidad de nombres de alumnos aprobados y reprobados
    print("Cantidad de nombres de alumnos aprobados:", contador_aprobados)
    print("Cantidad de nombres de alumnos reprobados:", contador_reprobados)

# Llama a la función main si este script es ejecutado directamente
if __name__ == "__main__":
    main()
