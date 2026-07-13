include(CheckCSourceCompiles)
include(CMakePushCheckState)

macro(CHECK_A64SVE2 VARIABLE)
    if(NOT DEFINED ${VARIABLE} OR NOT ${VARIABLE})
        message(STATUS "Check whether A64 SVE2 can be used")
        cmake_push_check_state(RESET)
        if(NOT MSVC)
            set(CMAKE_REQUIRED_FLAGS "-march=armv9-a+sve2")
        endif()
        # Use STATIC_LIBRARY to avoid link issues - we only need to verify compilation
        set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
        check_c_source_compiles("
            #include <arm_sve.h>
            int test_sve2(void) {
                svuint64_t a = svdup_u64(0x12345678ULL);
                svuint64_t b = svdup_u64(0x0F0F0F0FULL);
                /* Use base SVE2 operation (no bitperm required) */
                svbool_t pred = svptrue_b64();
                svuint64_t c = svorr_u64_x(pred, a, b);
                (void)c;
                return 0;
            }" HAVE_${VARIABLE})
        cmake_pop_check_state()
        if(HAVE_${VARIABLE})
            message(STATUS "Check whether A64 SVE2 can be used - yes")
            set(${VARIABLE} 1 CACHE INTERNAL "Result of CHECK_A64SVE2" FORCE)
        else()
            message(STATUS "Check whether A64 SVE2 can be used - no")
        endif()
    endif()
endmacro(CHECK_A64SVE2)