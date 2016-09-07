/* 
   CS288 HW10
	Point-to-point matrix multiplication using send/receive, bcast.
*/
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define NELMS 1000000
#define MASTER 0
#define MAXPROCS 16

void matrix_mult();
void init_lst();
void print_lst();

int n;

int main(int argc, char **argv) {
  int i,portion,elems,pid,nprocs, rank;
  double stime,etime;
  MPI_Status status;
  MPI_Comm world;

  n = atoi(argv[1]);
  if (n > NELMS) { printf("n=%d > N=%d\n",n,NELMS); exit(1); }

	long matrix_a[n][n], matrix_b[n][n], matrix_c[n][n];

  MPI_Init(&argc, &argv);
  world = MPI_COMM_WORLD;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
 
	portion = n/nprocs;
	elems = portion * n;

	if (pid == MASTER)
		init_lst(matrix_a, matrix_b, n);

	if(pid == MASTER){
		for(i = 1; i < nprocs; i++)
			MPI_Send(&matrix_a[i*portion][0], elems, MPI_LONG, i , 123, world);
	}
	else
		MPI_Recv(&matrix_a[pid * portion][0], elems, MPI_LONG, MASTER, 123, world, &status);

	MPI_Bcast(matrix_b, n*n, MPI_LONG, MASTER, world);

	stime = MPI_Wtime();

	matrix_mult(matrix_a, matrix_b, matrix_c, portion, pid);

	if(pid == MASTER){
		for(i = 1; i < nprocs; i++)
			MPI_Recv(&matrix_c[i * portion][0], elems, MPI_LONG, i , 456, world, &status);
	}
	else
		MPI_Send(&matrix_c[pid * portion][0], elems, MPI_LONG, MASTER,  456, world);

  etime = MPI_Wtime();

  if (pid == MASTER)
		printf("pid=%d: elapsed=%f\n",pid,etime-stime);

  MPI_Finalize();
}

void matrix_mult(long a[][n], long b[][n], long c[][n], int portion, int pid){
	int i, j, k;

	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++)
			c[i][j] = 0;
	}

	for(i = pid * portion; i  <  (pid+1)*portion; i++){
		for(j=0; j < n; j++){
			for(k = 0; k < n; k++){
				c[i][j] = c[i][j] + (a[i][k] * b[k][j]);
			}
		}
	}
}

void init_lst(long a[][n], long b[][n], int n){
  int i, j;
	
	//initializes matrix a
	for (i=0; i<n; i++) {
		for (j = 0; j < n; j++)
			a[i][j] = j;
	}
	//initializes matrix b
	for (i = 0; i<n; i++) {
		for (j=0 ; j < n; j++)
			b[i][j] = j;
	}
}

// end of file
