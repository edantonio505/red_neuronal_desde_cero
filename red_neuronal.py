import math
import random
import tqdm












# DESCARGAR ARCHIVOS A LA CARPETA
# mnist_dataset/
# set de datos de entrenamiento completo
# http://www.pjreddie.com/media/files/mnist_train.csv


# set de datos de testeo completo
# http://www.pjreddie.com/media/files/mnist_test.csv









# ===============================
# funciones de algebra lineal
# ==============================

# suma de vectors
def add(v, w):
    assert len(v) == len(w)
    return [v_i + w_i for v_i, w_i in zip(v, w)]


# resta de vectores
def subtract(v, w):
    assert len(v) == len(w)
    return [v_i - w_i for v_i, w_i in zip(v, w)]


# multiplication escalar
def scalar_multiply(c, v):
    return [v_i*c for v_i in v]



# mulitplication matricial
def dot(v, w):
    assert len(v) == len(w)
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

















# ========================
# fnciones de calculo
# ========================

def  sum_of_squares(v):
    return dot(v, v)




def squared_distance(v, w):
    return sum_of_squares(subtract(v, w))



# pado de gradiente
def gradient_step(v, gradient, step_size):
    step = scalar_multiply(step_size, gradient)
    return add(v, step)













# ========================
# funciones de Red Neuronal
# ========================

def sigmoid(t):
    return 1 / (1 + math.exp(-t))


# weights matriz
# input vectoir
# output neurona
def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))








# feed forward algorithm
def feed_forward(neural_network , input_vector):
    outputs = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]              # Add a constant.
        output = [neuron_output(neuron, input_with_bias)  # Compute the output
                  for neuron in layer]                    # for each neuron.
        outputs.append(output)                            # Add to results.        
        input_vector = output
    return outputs










# escupe el maximo valor de un vector
def argmax(xs: list) -> int:
    """Returns the index of the largest value"""
    return max(range(len(xs)), key=lambda i: xs[i])


















# funcion para convertir los 
# set de datos desde texto a vectores
def convertir_letras_en_vector(dataframe):
    new_dataframe = []
    new_labels = []
    for vector in dataframe:
        all_data = vector.split(",")
        data = all_data[1:] 
        new_labels.append(all_data[0])
        # normalizr entre 0.0 / 255.0 1.0
        new_dataframe.append([float(numero) / 255.0 * 0.99 + 0.01 for numero in data])  
    
    # """
    #     etiquetas
    #     ==========
    #     0 = [0.99, 0.1, 0.1, 0.1, 0.1 , 0.1, 0.1, 0.1, 0.1, 0.1]
    #     1 = [0.1, 0.99, 0.1, 0.1, 0.1 , 0.1, 0.1, 0.1, 0.1, 0.1]
    #     ...
    # """
    encoded_labels = []
    for numero in new_labels:
        label = [0.1 for _ in range(10)]
        label[int(numero)] = 0.99   
        encoded_labels.append(label)    

    return encoded_labels, new_dataframe















# funcion que convierte vector en numero 
# que se puede en la linea de comandos
def mostrar_numeros(vector, etiqueta):
    # 784
    linea = ""
    texto = ""
    contador = 0    
    for i in range(len(vector)):    
        numero = vector[i]
        if numero > 0.01:
            linea += "#"
        else:
            linea += " "

        if contador % 28 == 0:
            texto += linea + "\n"
            linea = ""
            contador = 0
        contador += 1
    print("el numero es: {}".format(argmax(etiqueta)))
    print(texto)











# Calculo + Feed_forward
# Finds the Gradient
# Calculus
def sqerror_gradients(network, input_vector, target_vector):
    # forward
    # feed forward
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # gradients with respect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]

    # gradients with respect to output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    # gradients with respect to hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                         dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]













# funcion para medir la precisioon del Red neuronal
def funcion_de_testeo(X_test, y_test, network):
    num_correct = 0
    for x, y in zip(X_test,  y_test):
        predicted = argmax(feed_forward(network, x)[-1])
        actual = argmax(y)
    
        if predicted == actual:
            num_correct += 1

        print("predicho : {}".format(predicted))
        mostrar_numeros(x, y)
    
        print("==================================")
        print()

    precision = int(num_correct * 100 / len(y_test))
    print("Esta red neuronal tiene una precision de: {}%".format(precision))


























def main():
    # cargar nuestros datos

    # datos de entrenamiento
    data_file = open("mnist_dataset/mnist_train.csv", "r")
    data_list = data_file.readlines()
    data_file.close()

    # etiquetas, y datos de entrenamiento
    y_train, X_train = convertir_letras_en_vector(data_list)



    # reducimos el tamano 
    # a un nivel manejable para python
    # y nuestra computadora
    tamano = 3000
    y_train = y_train[:tamano]
    X_train = X_train[:tamano]

    # indice = 2
    # mostrar_numeros(X_train[indice], y_train[indice])

    






    #====================================================== 
    # capas de input
    input_nodes = 784

    # capas ocultas
    hidden_nodes = 100

    # capas output
    output_nodes = 10

    # rango de aprendizaje
    # alpha
    learning_rate = 0.3

    # epocas
    epoch = 2
    #====================================================== 







    # crear red neuronal
    # con inicializacion aleatoria de los pesos
    network = [
        # capa oculta
        [
            [random.random() / 100 for _ in range(input_nodes + 1) ] for _ in range(hidden_nodes)
        ],

        # capa de salida
        [
            [random.random() / 100 for _ in range(hidden_nodes + 1)] for _ in range(output_nodes)
        ] 
    ]




    



    # entrenando el algoritmo
    # entrenando la red neuronal


    # entrenando la red neuronal==================================================
    print("Estamos entrenando la red neuronal")
    with tqdm.trange(epoch) as t:
        for epoch in t:
            epoch_loss = 0.0 

            for x, y in zip(X_train,  y_train):                
                predicted = feed_forward(network, x)[-1]
                epoch_loss += squared_distance(predicted, y)

                # calculamos la griente
                gradients = sqerror_gradients(network, x, y)
                # actualizamos los pesos de 
                # la red neuronal
                network = [[  gradient_step(neuron, grad, -learning_rate)
                            for neuron, grad in zip(layer, layer_grad)]
                        for layer, layer_grad in zip(network, gradients)]
    
            t.set_description(f"distancia de error: {epoch_loss:.2f}")
    # ==============================================================================


    


    



    # testeando la red neuronal
    # 

    data_file = open("mnist_dataset/mnist_test.csv", "r")
    data_list = data_file.readlines()
    data_file.close()

    y_test, X_test = convertir_letras_en_vector(data_list)


    tamano = 50
    y_test = y_test[:tamano]
    X_test = X_test[:tamano]

    # funcion para testear  
    # precision de red neuronal
    funcion_de_testeo(X_test, y_test, network)
   





if __name__ == "__main__":
    main()


