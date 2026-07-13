/*
 * Test program to determine which SVE2 sub-extensions are supported.
 * Compile with: clang -march=armv9-a+sve2-bitperm -o test_sve2_features test_sve2_features.c
 * Run: ./test_sve2_features
 *
 * If basic SVE2 works but bitperm crashes, your CPU supports SVE2 but NOT sve2-bitperm.
 */

#include <stdio.h>
#include <signal.h>
#include <setjmp.h>
#include <arm_sve.h>

static jmp_buf jump_buffer;
static volatile int got_signal = 0;

#ifdef _WIN32
#include <windows.h>

static LONG WINAPI exception_handler(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION) {
        got_signal = 1;
        longjmp(jump_buffer, 1);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#else
static void signal_handler(int sig) {
    got_signal = 1;
    longjmp(jump_buffer, 1);
}
#endif

/* Test 1: Basic SVE2 instruction (svmatch - part of base SVE2) */
__attribute__((noinline))
int test_basic_sve2(void) {
    /* Use a simple SVE2 instruction: MATCH (svmatch) */
    svuint8_t a = svdup_u8(0x42);
    svuint8_t b = svdup_u8(0x42);
    svbool_t pred = svptrue_b8();
    svbool_t result = svmatch_u8(pred, a, b);
    return (int)svptest_any(svptrue_b8(), result);
}

/* Test 2: SVE2-BITPERM instruction (svbext - requires sve2-bitperm) */
__attribute__((noinline))
int test_sve2_bitperm(void) {
    svuint64_t src = svdup_u64(0x12345678ULL);
    svuint64_t mask = svdup_u64(0x0F0F0F0FULL);
    svuint64_t result = svbext_u64(src, mask);
    return (int)svlastb_u64(svptrue_b64(), result);
}

int main(void) {
    printf("=== SVE2 Feature Detection ===\n");
    printf("CPU: Testing SVE2 sub-extensions...\n\n");

#ifdef _WIN32
    LPTOP_LEVEL_EXCEPTION_FILTER old_handler = SetUnhandledExceptionFilter(exception_handler);
#else
    struct sigaction sa, old_sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &old_sa);
#endif

    /* Test 1: Basic SVE2 */
    printf("Test 1: Basic SVE2 (svmatch)... ");
    fflush(stdout);
    got_signal = 0;
    if (setjmp(jump_buffer) == 0) {
        int r = test_basic_sve2();
        printf("PASSED (result=%d)\n", r);
    } else {
        printf("FAILED - Illegal instruction! SVE2 NOT supported.\n");
#ifdef _WIN32
        SetUnhandledExceptionFilter(old_handler);
#else
        sigaction(SIGILL, &old_sa, NULL);
#endif
        return 1;
    }

    /* Test 2: SVE2-BITPERM */
    printf("Test 2: SVE2-BITPERM (svbext)... ");
    fflush(stdout);
    got_signal = 0;
    if (setjmp(jump_buffer) == 0) {
        int r = test_sve2_bitperm();
        printf("PASSED (result=%d)\n", r);
    } else {
        printf("FAILED - Illegal instruction! sve2-bitperm NOT supported.\n");
        printf("\n>>> Your CPU supports SVE2 but NOT the bitperm sub-extension. <<<\n");
#ifdef _WIN32
        SetUnhandledExceptionFilter(old_handler);
#else
        sigaction(SIGILL, &old_sa, NULL);
#endif
        return 2;
    }

#ifdef _WIN32
    SetUnhandledExceptionFilter(old_handler);
#else
    sigaction(SIGILL, &old_sa, NULL);
#endif

    printf("\nAll tests passed! SVE2 + bitperm are both supported.\n");
    return 0;
}