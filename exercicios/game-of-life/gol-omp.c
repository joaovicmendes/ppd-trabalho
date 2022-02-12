/* 
 ** Universidade Federal de S?o Carlos
 ** Departamento de Computa??o
 ** Prof. H?lio Crestana Guardia
 ** Programa??o Paralela e Distribu?da
 */

/*
 * The (Conway's) Game of Life
 *
 * Any live cell with fewer than two live neighbours dies, as if by underpopulation.
 * Any live cell with two or three live neighbours lives on to the next generation.
 * Any live cell with more than three live neighbours dies, as if by overpopulation.
 * Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

 * O espaço usado nesta versão do jogo é uma matriz bidimensional, sendo que as 
 * bordas nas laterais e superior e inferior estão logicamente conectadas. Deste modo,
 * tem-se o comportamento de um toroide [https://pt.wikipedia.org/wiki/Toro_(topologia)].

 * Em cada célula, o valor 1 indica que ela está viva, e 0 que está morta.
 *
 * Como parâmetros de entrada devem ser fornecidos a dimensão (linhas = colunas), o
 * número de iterações e os respectivos dados. É claro que os dados de entrada do 
 * programa podem ser obtidos de um arquivo, redirecionando-se STDIN na execução.
 *
 * Ao ler dados de entrada das células, 'x' indica célula viva e ' ' indica célula morta.
 * 
 * A geração da população inicial também pode ocorrer de maneira aleatória, fornecendo-se
 * apenas as dimensões e o número de iterações.
 */

// Adaptado de 
// http://lspd.mackenzie.br/marathon/16/problems.html
// https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
// https://rosettacode.org/wiki/Conway%27s_Game_of_Life#C

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

// #define DEBUG

typedef unsigned char cell_t; 


// número de células vizinhas de i,j 
int neighbors (cell_t *board, int size, int i, int j) 
{
	int count;

	count = board [   i                 * size + ( (j-1+size) % size) ] +
	        board [   i                 * size + ( (j+1) % size     ) ] +
	        board [ ((i-1+size) % size) * size +    j                 ] +
	        board [ ((i-1+size) % size) * size + ( (j-1+size) % size) ] +
	        board [ ((i-1+size) % size) * size + ( (j+1) % size     ) ] +
	        board [ ((i+1)      % size) * size +    j                 ] +
	        board [ ((i+1)      % size) * size + ( (j-1+size) % size) ] +
	        board [ ((i+1)      % size) * size + ( (j+1) % size     ) ];

	return count;
}

// analisa as células da população atual e suas vizinhanças 
void play (cell_t *board, cell_t *newboard, int size) 
{
	int i, j, n;

	// for each cell, apply the rules of Life
	#pragma omp for
	for (i=0; i < size; i++)
		for (j=0; j < size; j++) {
			n = neighbors (board, size, i, j);
			if (n == 2) newboard [i*size+j] = board[i*size+j];
			if (n == 3) newboard [i*size+j] = 1;
			if (n < 2) newboard [i*size+j] = 0;
			if (n > 3) newboard [i*size+j] = 0; 
		}
}

// exibe o estado da população: arte ASCII no terminal :-)
void show (cell_t *board, int size) 
{
	for (int y=0; y < size; y++) {
		for (int x=0; x < size; x++) {
			// printf("\033[%d;%df",y+1,2*x+1); // ajusta posição do cursor
			// \033[07m: cursor invertido (p pintar); \033[m: default (p apagar)
			// printf(board[y*size+x] ? "\033[07m  " : "\033[m  ");  

			// ajusta posição e imprime: melhor escrever numa única operação
			printf( board[y*size+x] ? "\033[%d;%df\033[07m  " : "\033[%d;%df\033[m  ", y+1, 2*x+1);
		}
	}
	fflush(stdout);	// força saída na tela
}

// lê configuração da população de STDIN
void read_board (cell_t * board, int size) 
{
	int  i, j;
	char c;

	for (i=0; i < size; i++) {
		for (j=0; j < size; j++ ) {
		 	do {
				c=getchar();
			} while(c=='\n');
			// putchar(c);  // se quiser conferir :-)

			board [i*size+j] = c == 'x';

			// printf("%d ",board[i*size+j]);  // se quiser conferir vendo a população
		}
		// printf("\n"); fflush(stdout);
	}
}

// para ler um valor inteiro, byte a byte, de STDIN
int le_int()
{
	int i = 0;
	char c;
	char digit[32];
	
	for (;;) {
		c = getchar();
		if	( c==' ' || c=='\n' || c=='\0' || i==31 )
			break;
		digit[i++] = c;
	}
	digit[i] = '\0';
	
	return atoi(digit);
} 


int main () 
{
	int size, steps;
	int i,j;

	size = le_int();
	steps = le_int();

	// printf("size: %d  steps: %d\n", size, steps);

	cell_t * prev = (cell_t *) malloc ( size * size * sizeof(cell_t) );
	cell_t * next = (cell_t *) malloc ( size * size * sizeof(cell_t) );
	cell_t * tmp;

// ([des]comente) para ler dados de STDIN ou gerá-los aleatoriamente
#define READ_FILE

#ifdef READ_FILE
	// Lê dados de stdin
	read_board (prev, size);
#else
	// Gera dados aleatoriamente
	srand(time(NULL));
	for (int i=0; i < size; i++)
		for (int j=0; j < size; j++)
			prev [i*size+j] = rand() < RAND_MAX / 10 ? 1 : 0;
#endif

#ifdef DEBUG
	printf("\033[%du\033%dt",size+1,size+1); // ajusta o tamanho da janela do terminal
	printf("\033[H\033[J");    // posiciona cursor no canto superior esquerdo e limpa a tela

	show(prev,size);

	sleep(1); // dorme 1 s, se quiser ver a população inicial
#endif

	for (i=0; i < steps; i++) {
		#pragma omp parallel
		play (prev,next,size);

#ifdef DEBUG
		// Descomente aqui e comente em "play" para exibir 
		// show(next,size);
		usleep(150000);
#endif
		tmp = next;
		next = prev;
		prev = tmp;
	}

#ifdef DEBUG
	// show(prev,size);
#endif

	free(prev);
	free(next);
}

