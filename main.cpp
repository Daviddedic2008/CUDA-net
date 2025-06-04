// cpp main file used to invoke kernel calls

#include "main.h"
#include "kernel.h"

using namespace std;

int main()
{
	launchKernel(10, 10);

	cout << "Hello CMake." << endl;
	return 0;
}
