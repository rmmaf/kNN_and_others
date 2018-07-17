import matplotlib.pyplot as plt
eixoX = [0.07142857142857142, 0.21428571428571427, 0.3571428571428571, 0.5, 0.6428571428571428, 0.7857142857142856, 0.9285714285714284, 1.0]
desvio = [0.025369827771028294, 0.11748731508611449, 0.2603444579432574, 0.40320160080040024, 0.546058743657543, 0.6889158865146859, 0.9746301722289716, 0.0]
eixoYK = [100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0]
eixoYA = [100.0, 100.0, 97.36842105263158, 81.81818181818183, 68.18181818181817, 50.0, 0.0, 0.0]
eixoYW = [100.0, 100.0, 97.36842105263158, 81.81818181818183, 59.09090909090909, 25.0, 0.0, 0.0]

plt.errorbar(eixoX, eixoYA, desvio)
plt.errorbar(eixoX, eixoYK, desvio)
plt.errorbar(eixoX, eixoYW, desvio)

plt.axis([0.00, 1.00, 0.00, 100.00])
plt.savefig('grafico.pdf', dpi = 600)
plt.show()

