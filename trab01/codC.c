int *saxpy(int N) {
    N = 1000;
    int x[N], *y[N];
    int a = 2;
    for(int i=0; i<N; i++){
        x[i] = i;
        *y[i] = i;
    }
    for (int i=0; i < N; i++)
        y[i] = a*x[i] + y[i];
    return *y;
}