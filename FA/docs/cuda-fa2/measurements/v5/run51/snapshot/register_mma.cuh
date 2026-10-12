#pragma once

// Keep the complete 8x8 update in one PTX dependency graph. ptxas still assigns
// physical registers and schedules SASS; this does not promise fixed registers.
#define FA2_FMA_PAIR(C00, C01, C10, C11, A0, A1, B0, B1) \
    "fma.rn.f32 %" #C00 ", %" #A0 ", %" #B0 ", %" #C00 ";\n" \
    "fma.rn.f32 %" #C10 ", %" #A1 ", %" #B0 ", %" #C10 ";\n" \
    "fma.rn.f32 %" #C11 ", %" #A1 ", %" #B1 ", %" #C11 ";\n" \
    "fma.rn.f32 %" #C01 ", %" #A0 ", %" #B1 ", %" #C01 ";\n"

#define FA2_ACCUMULATOR_ROW(R) \
    "+f"(c[R][0]), "+f"(c[R][1]), "+f"(c[R][2]), "+f"(c[R][3]), \
    "+f"(c[R][4]), "+f"(c[R][5]), "+f"(c[R][6]), "+f"(c[R][7])

__device__ __forceinline__ void register_mma_8x8(
    float (&c)[8][8], const float (&a)[8], const float (&b)[8]) {
    asm(
        FA2_FMA_PAIR(16, 17, 0, 1, 66, 64, 72, 73)
        FA2_FMA_PAIR(24, 25, 8, 9, 67, 65, 72, 73)
        FA2_FMA_PAIR(48, 49, 32, 33, 70, 68, 72, 73)
        FA2_FMA_PAIR(56, 57, 40, 41, 71, 69, 72, 73)
        FA2_FMA_PAIR(58, 59, 42, 43, 71, 69, 74, 75)
        FA2_FMA_PAIR(50, 51, 34, 35, 70, 68, 74, 75)
        FA2_FMA_PAIR(26, 27, 10, 11, 67, 65, 74, 75)
        FA2_FMA_PAIR(18, 19, 2, 3, 66, 64, 74, 75)
        FA2_FMA_PAIR(20, 21, 4, 5, 66, 64, 76, 77)
        FA2_FMA_PAIR(28, 29, 12, 13, 67, 65, 76, 77)
        FA2_FMA_PAIR(52, 53, 36, 37, 70, 68, 76, 77)
        FA2_FMA_PAIR(60, 61, 44, 45, 71, 69, 76, 77)
        FA2_FMA_PAIR(62, 63, 46, 47, 71, 69, 78, 79)
        FA2_FMA_PAIR(54, 55, 38, 39, 70, 68, 78, 79)
        FA2_FMA_PAIR(30, 31, 14, 15, 67, 65, 78, 79)
        FA2_FMA_PAIR(22, 23, 6, 7, 66, 64, 78, 79)
        : FA2_ACCUMULATOR_ROW(0), FA2_ACCUMULATOR_ROW(1),
          FA2_ACCUMULATOR_ROW(2), FA2_ACCUMULATOR_ROW(3),
          FA2_ACCUMULATOR_ROW(4), FA2_ACCUMULATOR_ROW(5),
          FA2_ACCUMULATOR_ROW(6), FA2_ACCUMULATOR_ROW(7)
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(a[4]), "f"(a[5]), "f"(a[6]), "f"(a[7]),
          "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
          "f"(b[4]), "f"(b[5]), "f"(b[6]), "f"(b[7]));
}

#undef FA2_ACCUMULATOR_ROW
#undef FA2_FMA_PAIR
