/**
 * Simplified Step 1 test to diagnose the issue
 */

#include <cstdio>
#include <cstdlib>
#include "../include/test_utils.h"

int main() {
    printf("Starting test...\n");
    fflush(stdout);

    printf("Calling print_device_info...\n");
    fflush(stdout);
    
    print_device_info();
    
    printf("print_device_info completed!\n");
    fflush(stdout);

    printf("Test completed successfully!\n");
    return 0;
}
