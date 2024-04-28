import cv2 
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt 


def cargar_imagenes(rutas):
    imagenes_grises = []
    for ruta in rutas:
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        imagenes_grises.append(img)
        mostrar_imagen(img, title=ruta)
    return imagenes_grises

def mostrar_imagen(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

# Cargar las imágenes de los exámenes en escala de grises y mostrarlas
rutas_examenes = ['img/multiple_choice_1.png', 'img/multiple_choice_2.png', 
                  'img/multiple_choice_3.png', 'img/multiple_choice_4.png', 
                  'img/multiple_choice_5.png']

examenes = cargar_imagenes(rutas_examenes)

def campos_encabezado_y_multiple_choice(imagen_bool):
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

    nombre, legajo, tipo, dia, multiple_choice = campos_encabezado_y_multiple_choice(examen)

    diccionario_validacion = {
        "Nombre y Apellido": validar_nombre(nombre),
        "Legajo": validar_legajo(legajo),
        "Código": validar_codigo(tipo),
        "Fecha": validar_fecha(dia)
    }

    return diccionario_validacion, multiple_choice, nombre


def respuestas_examen(multiple_choice_binario):
    respuestas = []
    preguntas = []
    img_rows = np.sum(multiple_choice_binario, 1)

    umbral_columna_valida = 20
    columnas_validas = img_rows >= umbral_columna_valida
    cambios = np.diff(columnas_validas)    
    indices_renglones = np.argwhere(cambios)    
    indices_renglones = indices_renglones[1:]
    indices_renglones = indices_renglones.reshape((-1,2))     

    for i in range(len(indices_renglones)):
        renglon = multiple_choice_binario[indices_renglones[i][0]:indices_renglones[i][1], 0:1000]    
        
        # Convertir a binario para Hugues Transform
        imagen_binaria = renglon.astype(np.uint8) * 255

        # Aplicar la transformada de Hough circular
        circulos = cv2.HoughCircles(imagen_binaria, cv2.HOUGH_GRADIENT, dp=2, minDist=20, param1=50, param2=20, minRadius=10, maxRadius=20)

        # Ordenar por su valor en el eje X antes de revisar si están
        circulos = sorted(circulos[0, :], key=lambda circle: circle[0])
        
        # Redondeo de coordenadas para el bucle  
        circulos = np.uint16(np.around(circulos))

        # Enumeración con una 4ta dimensión    
        circulos = [np.append(circle, x + 1) for x, circle in enumerate(circulos)]    
        
        circulos = np.array(circulos)
        
        # Crear una copia de la imagen original para dibujar los círculos
        imagen_salida = cv2.cvtColor(imagen_binaria, cv2.COLOR_GRAY2BGR)
        
        tamaño_vecindad = 10

        cuenta = 0 

        for circulo in circulos:
            centro = (circulo[0], circulo[1])
            radio = circulo[2]
            # Dibujar el círculo en verde
            cv2.circle(imagen_salida, centro, radio, (0, 255, 0), 2)
            # Verificar si al menos una cantidad mínima de píxeles dentro del círculo están rellenos de blanco (255)
            x, y = circulo[0], circulo[1]
            vecindad_x = slice(max(0, x - tamaño_vecindad), min(imagen_binaria.shape[1], x + tamaño_vecindad + 1))
            vecindad_y = slice(max(0, y - tamaño_vecindad), min(imagen_binaria.shape[0], y + tamaño_vecindad + 1))
            píxeles_vecindad = imagen_binaria[vecindad_y, vecindad_x]
            está_relleno = np.count_nonzero(píxeles_vecindad == 255) >= píxeles_vecindad.size * 0.5  # Al menos el 50% de la vecindad debe estar rellena
            # Si el círculo está relleno de blanco, dibujar un contorno rojo
            if está_relleno:
                guarda_pregunta = np.append(preguntas, i+1)
                guarda_respuesta = np.append(respuestas, circulo[3])
                cv2.circle(imagen_salida, centro, radio, (0, 0, 255), 2)
                cuenta += 1
            
        if cuenta == 0 or cuenta > 1:
            preguntas = np.append(preguntas, i+1)
            respuestas = np.append(respuestas, 0)
        else:
            preguntas = guarda_pregunta
            respuestas = guarda_respuesta

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



resultados_concat = None  

for indice, examen in enumerate(examenes):
    img_th = examen < 200   
    dic_resultados_validacion, imagen_multi_choice, nombre = validar_campos(img_th)
    dic_respuestas_preguntas = respuestas_examen(imagen_multi_choice)
    dic_examen_corregido, aprobado = corregir_examen(dic_respuestas_preguntas)

    if aprobado:
        nombre_resultado = 255 - nombre  
    else:
        nombre_resultado = 255 + nombre

    if resultados_concat is None:
        resultados_concat = nombre_resultado
    else:
        resultados_concat = np.vstack((resultados_concat, nombre_resultado))

    # Impresión de resultados
    print("+------------------------+-----------+")
    print(f"|         EXAMEN {indice+1}                   |")
    print("+------------------------+-----------+")
    print("| Validación Campos      | Resultado |")
    print("+------------------------+-----------+")
        
    for campo, resultado in dic_resultados_validacion.items():
        print(f"| {campo:<22} |   {resultado:<7} |")
        print("+------------------------+-----------+")

    print("+------------------------+-----------+")
    print("|           Corrección               |")
    print("+------------------------+-----------+")
    print("| Preguntas              | Resultado |")
    print("+------------------------+-----------+")

    for pregunta, resultado in dic_examen_corregido.items():
        print(f"| {int(pregunta):<22} |   {resultado:<7} |")
        print("+------------------------+-----------+")

    print("")
    input("PRESIONE ENTER PARA EVALUAR EL PRÓXIMO EXAMEN...")
    print("")


# Normalización de los valores para que puedan ser guardados en una imagen
min_val = np.min(resultados_concat)
max_val = np.max(resultados_concat)
resultados_normalizados = 255 * (resultados_concat - min_val) / (max_val - min_val)
resultados_normalizados = resultados_normalizados.astype(np.uint8)

# Guardar la imagen
cv2.imwrite('nota_examen.png', resultados_normalizados)

print("La imagen nota_examen.png contiene aquellos alumnos que han aprobado y aquellos que han reprobado")
print("El alumno ha aprobado si en la imagen nota_examen.png el nombre aparece en negro")
print("El alumno ha reprobado si en la imagen nota_examen.png el nombre aparece en blanco")
print("")

# Se muestra la imagen de alumnos aprobados (nombre en negro) y reprobados (nombre en blanco)

plt.imshow(resultados_normalizados, cmap='gray')
plt.show()


