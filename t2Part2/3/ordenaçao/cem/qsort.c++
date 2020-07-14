#include <iostream>
#include <fstream>
using namespace std;
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
    long long int tot = 100;
    cout << tot << endl;
    int *val = new int[tot];
    for(long long int i=0;i<tot;i++){
        cin >> val[i];
    }
   //ordena o array
   qsort(val, tot, sizeof(int), comparador);

    //mostra os valores do array
   /*for(long long int i = 0 ; i < tot; i++ ) {
      cout << val[i] << " ";
   }
   cout << endl;*/
   return 0;
}
