int add( int a ,int b ){return a+b;} 

int divide(int a, int b) {          // BUG: no divide-by-zero check
    return a / b;
}
