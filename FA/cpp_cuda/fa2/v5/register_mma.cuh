#pragma once

// Keep the complete 8x8 update in one PTX dependency graph. ptxas still assigns
// physical registers and schedules SASS; this does not promise fixed registers.
#define FA2_FMA(C, A, B) \
    "fma.rn.f32 %" #C ", %" #B ", %" #A ", %" #C ";\n"

#define FA2_COLUMN_UP(C0, C1, C2, C3, C4, C5, C6, C7, B) \
    FA2_FMA(C0, 64, B) FA2_FMA(C1, 65, B) FA2_FMA(C2, 66, B) FA2_FMA(C3, 67, B) \
    FA2_FMA(C4, 68, B) FA2_FMA(C5, 69, B) FA2_FMA(C6, 70, B) FA2_FMA(C7, 71, B)

#define FA2_COLUMN_DOWN(C0, C1, C2, C3, C4, C5, C6, C7, B) \
    FA2_FMA(C7, 71, B) FA2_FMA(C6, 70, B) FA2_FMA(C5, 69, B) FA2_FMA(C4, 68, B) \
    FA2_FMA(C3, 67, B) FA2_FMA(C2, 66, B) FA2_FMA(C1, 65, B) FA2_FMA(C0, 64, B)

#define FA2_ACCUMULATOR_ROW(R) \
    "+f"(c[R][0]), "+f"(c[R][1]), "+f"(c[R][2]), "+f"(c[R][3]), \
    "+f"(c[R][4]), "+f"(c[R][5]), "+f"(c[R][6]), "+f"(c[R][7])

__device__ __forceinline__ void register_mma_8x8(
    float (&c)[8][8], const float (&a)[8], const float (&b)[8]) {
    asm(
        FA2_COLUMN_UP(0, 8, 16, 24, 32, 40, 48, 56, 72)
        FA2_COLUMN_DOWN(1, 9, 17, 25, 33, 41, 49, 57, 73)
        FA2_COLUMN_UP(2, 10, 18, 26, 34, 42, 50, 58, 74)
        FA2_COLUMN_DOWN(3, 11, 19, 27, 35, 43, 51, 59, 75)
        FA2_COLUMN_UP(4, 12, 20, 28, 36, 44, 52, 60, 76)
        FA2_COLUMN_DOWN(5, 13, 21, 29, 37, 45, 53, 61, 77)
        FA2_COLUMN_UP(6, 14, 22, 30, 38, 46, 54, 62, 78)
        FA2_COLUMN_DOWN(7, 15, 23, 31, 39, 47, 55, 63, 79)
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
#undef FA2_COLUMN_DOWN
#undef FA2_COLUMN_UP
#undef FA2_FMA
