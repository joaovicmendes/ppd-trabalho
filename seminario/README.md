# Seminário - Yet More Primes
Seminário sobre paralelização do algoritmo Yet More Primes, da 15a Maratona de Programação Paralela.

# Grupo
- [Guilherme Lorençato Lamonato](https://github.com/GuiLorencato), 758665
- [João Victor Mendes Freire](https://github.com/joaovicmendes), 758943
- [Julia Cinel Chagas](https://github.com/jcinel), 759314
- [Reynold Navarro Mazo](https://github.com/reynold125), 756188

# Resultado
|           |Serial |OpenMP |MPI    |CUDA|OpenMP + MPI|MPI + CUDA|
|:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|
|**input.1**|0.008	|*0.004*	|0.217	|0.310	|0.235	|0.691|
|**input.2**|1.291	|*0.604*	|0.977	|1.737	|0.909	|2.383|
|**input.3**|88.315	|37.805	|37.956	|*2.100*	|38.145	|4.346|
