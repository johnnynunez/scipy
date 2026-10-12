typedef struct { double real, imag; } complex_double;
int check_c(double r, const complex_double *z) {
    return r == 2.0 && z->real == 2.0 && z->imag == -3.0;
}
