#include <stdio.h>
#include <stdlib.h>

int comparador(const void *a, const void *b) {
   return ( *(int*)a - *(int*)b );
}
 
int comparador2(const void *a, const void *b) {
   if (*(int*)a > *(int*)b) {
      return 1;
   } else if (*(int*)a < *(int*)b) {
      return -1;
   } else {
      return 0;
   }
}
 
int main (int argc, char *argv[]) {
    
    int tot;
    scanf("%i", &tot);
    int val[tot];
    FILE *texiste;

    if(tot == 10){
        texiste = fopen("dez.txt", "r");
    }else if(tot == 100){
        texiste = fopen("cem.txt", "r");
    }else if(tot == 1000){
        texiste = fopen("mil.txt", "r");
    }else if(tot == 10000){
        texiste = fopen("dezMil.txt", "r");
    }else if(tot == 1000000){
        texiste = fopen("milhao.txt", "r");
    }
    for(int i=0;i<tot;i++){
        fscanf(texiste, "%i",&val[i]);
    }
 
   //ordena o array
   qsort(val, tot, sizeof(int), comparador);

    //mostra os valores do array
   for(int i = 0 ; i < tot; i++ ) {
      printf("%i ", val[i]);
   }
   printf("\n");
   fclose(texiste);
   return(0);
}