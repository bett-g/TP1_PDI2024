import cv2 
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt 
import matplotlib.patches as mpatches

def cargar_img(rutas):
    imagenes = []
    for ruta in rutas:
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        imagenes.append(img)
    return imagenes

def procesar_form(imagen_bool):
    img_rows = np.sum(imagen_bool, 1)
    filas_validas = img_rows > 500
    indices_validos = np.where(filas_validas)[0]
    umbral_espacio = 22  

    for i in range(1, len(indices_validos)):
        if indices_validos[i] - indices_validos[i - 1] > umbral_espacio:
            primer_espacio_index = i
            break

    indice_inicial = indices_validos[primer_espacio_index - 1] + 1
    indice_final = indices_validos[primer_espacio_index] - 1
    img_final = imagen_bool[indice_inicial:indice_final]

    img_cols = np.sum(img_final, 0)
    umbral_columna_valida = 19
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
        for i in range(2,num_labels):
            fila_actual = stats_ord[i]
            fila_anterior = stats_ord[i - 1]
            suma = fila_actual[0] - (fila_anterior[0] + fila_anterior[2])        
            resultados.append(suma)    
        espacios = 0
        for valor in resultados:
            if valor >= 5:
                espacios += 1
        palabras = espacios + 1
        caracteres = num_labels + espacios - 1
    return caracteres, palabras

def validar_campos(examen):  
    def validar_nombre(nombre):
        caracteres, palabras = contar_componentes_conectados(nombre)
        resultado_nombre = "OK" if 1 < caracteres <= 25 and palabras > 1 else "MAL"
        return resultado_nombre

    def validar_legajo(legajo):
        caracteres, palabras = contar_componentes_conectados(legajo)
        resultado_legajo = "OK" if caracteres == 8 and palabras == 1 else "MAL"
        return resultado_legajo

    def validar_codigo(tipo):
        caracteres, palabras = contar_componentes_conectados(tipo)
        resultado_codigo = "OK" if caracteres == 1 else "MAL"
        return resultado_codigo 

    def validar_fecha(dia):
        caracteres, palabras = contar_componentes_conectados(dia)
        resultado_fecha = "OK" if caracteres == 8 and palabras == 1 else "MAL"
        return resultado_fecha

    nombre, legajo, tipo, dia, multiple_choice = procesar_form(examen)

    diccionario_validacion = {
        "Nombre y Apellido": validar_nombre(nombre),
        "Legajo": validar_legajo(legajo),
        "Código": validar_codigo(tipo),
        "Fecha": validar_fecha(dia)
    }

    return diccionario_validacion, multiple_choice, nombre

def respuestas(multiple_choice_binario):
    respuestas = []  # Lista para almacenar las respuestas encontradas
    preguntas = []   # Lista para almacenar los números de pregunta correspondientes a las respuestas encontradas
    img_rows = np.sum(multiple_choice_binario, 1)  # Calcula el número de píxeles blancos en cada fila
    umbral_columna_valida = 20  # Define el umbral mínimo de píxeles blancos por fila para considerarla válida
    columnas_validas = img_rows >= umbral_columna_valida  # Encuentra las filas válidas basadas en el umbral definido
    cambios = np.diff(columnas_validas)  # Encuentra los cambios entre filas válidas e inválidas
    indices_renglones = np.argwhere(cambios)  # Encuentra los índices de cambio entre filas válidas e inválidas
    indices_renglones = indices_renglones[1:]  # Descarta el primer cambio, ya que no indica un inicio de pregunta
    indices_renglones = indices_renglones.reshape((-1,2))  # Agrupa los índices de cambio en pares representando rangos de filas

    # Itera sobre los rangos de filas válidas donde podría haber preguntas
    for i in range(len(indices_renglones)):
        renglon = multiple_choice_binario[indices_renglones[i][0]:indices_renglones[i][1], 0:1000]  # Recorta la región de interés de la imagen
        imagen_binaria = renglon.astype(np.uint8) * 255  # Convierte la región recortada en una imagen binaria
        circulos = cv2.HoughCircles(imagen_binaria, cv2.HOUGH_GRADIENT, dp=2, minDist=20, param1=50, param2=20, minRadius=10, maxRadius=20)  # Detecta círculos en la imagen binaria
        circulos = sorted(circulos[0, :], key=lambda circle: circle[0])  # Ordena los círculos detectados por su posición en el eje X
        circulos = np.uint16(np.around(circulos))  # Redondea las coordenadas de los círculos detectados
        circulos = [np.append(circle, x + 1) for x, circle in enumerate(circulos)]  # Agrega un identificador de pregunta a cada círculo detectado
        circulos = np.array(circulos)  # Convierte la lista de círculos a un array de NumPy

        imagen_salida = cv2.cvtColor(imagen_binaria, cv2.COLOR_GRAY2BGR)  # Convierte la imagen binaria en color para dibujar sobre ella
        tamaño_vecindad = 10  # Define el tamaño de la vecindad para comprobar si el círculo está relleno
        cuenta = 0  # Contador de círculos rellenos dentro de una pregunta

        # Itera sobre los círculos detectados en la imagen binaria
        for circulo in circulos:
            centro = (circulo[0], circulo[1])  # Coordenadas del centro del círculo
            radio = circulo[2]  # Radio del círculo
            cv2.circle(imagen_salida, centro, radio, (0, 255, 0), 2)  # Dibuja el círculo detectado en verde sobre la imagen de salida
            x, y = circulo[0], circulo[1]  # Coordenadas del centro del círculo
            vecindad_x = slice(max(0, x - tamaño_vecindad), min(imagen_binaria.shape[1], x + tamaño_vecindad + 1))  # Define la región en el eje X alrededor del centro del círculo
            vecindad_y = slice(max(0, y - tamaño_vecindad), min(imagen_binaria.shape[0], y + tamaño_vecindad + 1))  # Define la región en el eje Y alrededor del centro del círculo
            píxeles_vecindad = imagen_binaria[vecindad_y, vecindad_x]  # Extrae la vecindad de píxeles alrededor del centro del círculo
            está_relleno = np.count_nonzero(píxeles_vecindad == 255) >= píxeles_vecindad.size * 0.5  # Verifica si al menos el 50% de los píxeles en la vecindad están rellenos de blanco
            if está_relleno:  # Si el círculo está relleno de blanco
                guarda_pregunta = np.append(preguntas, i+1)  # Guarda el número de pregunta
                guarda_respuesta = np.append(respuestas, circulo[3])  # Guarda el identificador de respuesta
                cv2.circle(imagen_salida, centro, radio, (0, 0, 255), 2)  # Dibuja el contorno del círculo en rojo sobre la imagen de salida
                cuenta += 1  # Incrementa el contador de círculos rellenos
        
        # Determina la respuesta para la pregunta actual
        if cuenta == 0 or cuenta > 1:  # Si no se detecta ningún círculo relleno o se detectan más de uno
            preguntas = np.append(preguntas, i+1)  # Guarda el número de pregunta como no respondida (0)
            respuestas = np.append(respuestas, 0)
        else:  # Si se detecta exactamente un círculo relleno
            preguntas = guarda_pregunta  # Guarda el número de pregunta
            respuestas = guarda_respuesta  # Guarda el identificador de respuesta
        
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

rutas_examenes = ['C:\\Users\\betsa\\OneDrive\\Escritorio\\TUIA2024\\PDI\\TP\\TP1_PDI2024\\img\\multiple_choice_1.png', 
                  'C:\\Users\\betsa\\OneDrive\\Escritorio\\TUIA2024\\PDI\\TP\\TP1_PDI2024\\img\\multiple_choice_2.png', 
                  'C:\\Users\\betsa\\OneDrive\\Escritorio\\TUIA2024\\PDI\\TP\\TP1_PDI2024\\img\\multiple_choice_3.png', 
                  'C:\\Users\\betsa\\OneDrive\\Escritorio\\TUIA2024\\PDI\\TP\\TP1_PDI2024\\img\\multiple_choice_4.png', 
                  'C:\\Users\\betsa\\OneDrive\\Escritorio\\TUIA2024\\PDI\\TP\\TP1_PDI2024\\img\\multiple_choice_5.png']

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

    # Impresión de resultados sin guiones
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
resultados_colores[resultados_colores == [255, 255, 255]] = [255, 255, 255]

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
