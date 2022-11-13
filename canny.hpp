#include <iostream>

using namespace std;
#include "ap_fixed.h"
typedef int DTYPE;
typedef ap_fixed<43,20>  Float;
void canny(DTYPE* src, DTYPE* dst, int upperThresh, int lowerThresh);
