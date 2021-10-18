#include <memory.h>
#include <math.h>
#include <stdlib.h>
#include <emmintrin.h>
#include "omp.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

// Mirror data
int GetMirrorPos(int Length, int Pos)
{
    if (Pos < 0)
    {
        Pos = -Pos;
        while (Pos > Length)
            Pos -= Length;
    }
    else if (Pos >= Length)
    {
        Pos = Length - (Pos - Length + 2);
        while (Pos < 0)
            Pos += Length;
    }
    return Pos;
}

void GetOffsetPos(int *Pos, int Length, int Left, int Right)
{
    for (int X = -Left; X < Length + Right; X++)
    {
        Pos[X + Left] = GetMirrorPos(Length, X);
    }
}

void HistgramAddShort(unsigned short *X, unsigned short *Y)
{
    *(__m128i *)(Y + 0) = _mm_add_epi16(*(__m128i *)&Y[0], *(__m128i *)&X[0]);
    *(__m128i *)(Y + 8) = _mm_add_epi16(*(__m128i *)&Y[8], *(__m128i *)&X[8]);
    *(__m128i *)(Y + 16) = _mm_add_epi16(*(__m128i *)&Y[16], *(__m128i *)&X[16]);
    *(__m128i *)(Y + 24) = _mm_add_epi16(*(__m128i *)&Y[24], *(__m128i *)&X[24]);
    *(__m128i *)(Y + 32) = _mm_add_epi16(*(__m128i *)&Y[32], *(__m128i *)&X[32]);
    *(__m128i *)(Y + 40) = _mm_add_epi16(*(__m128i *)&Y[40], *(__m128i *)&X[40]);
    *(__m128i *)(Y + 48) = _mm_add_epi16(*(__m128i *)&Y[48], *(__m128i *)&X[48]);
    *(__m128i *)(Y + 56) = _mm_add_epi16(*(__m128i *)&Y[56], *(__m128i *)&X[56]);
    *(__m128i *)(Y + 64) = _mm_add_epi16(*(__m128i *)&Y[64], *(__m128i *)&X[64]);
    *(__m128i *)(Y + 72) = _mm_add_epi16(*(__m128i *)&Y[72], *(__m128i *)&X[72]);
    *(__m128i *)(Y + 80) = _mm_add_epi16(*(__m128i *)&Y[80], *(__m128i *)&X[80]);
    *(__m128i *)(Y + 88) = _mm_add_epi16(*(__m128i *)&Y[88], *(__m128i *)&X[88]);
    *(__m128i *)(Y + 96) = _mm_add_epi16(*(__m128i *)&Y[96], *(__m128i *)&X[96]);
    *(__m128i *)(Y + 104) = _mm_add_epi16(*(__m128i *)&Y[104], *(__m128i *)&X[104]);
    *(__m128i *)(Y + 112) = _mm_add_epi16(*(__m128i *)&Y[112], *(__m128i *)&X[112]);
    *(__m128i *)(Y + 120) = _mm_add_epi16(*(__m128i *)&Y[120], *(__m128i *)&X[120]);
    *(__m128i *)(Y + 128) = _mm_add_epi16(*(__m128i *)&Y[128], *(__m128i *)&X[128]);
    *(__m128i *)(Y + 136) = _mm_add_epi16(*(__m128i *)&Y[136], *(__m128i *)&X[136]);
    *(__m128i *)(Y + 144) = _mm_add_epi16(*(__m128i *)&Y[144], *(__m128i *)&X[144]);
    *(__m128i *)(Y + 152) = _mm_add_epi16(*(__m128i *)&Y[152], *(__m128i *)&X[152]);
    *(__m128i *)(Y + 160) = _mm_add_epi16(*(__m128i *)&Y[160], *(__m128i *)&X[160]);
    *(__m128i *)(Y + 168) = _mm_add_epi16(*(__m128i *)&Y[168], *(__m128i *)&X[168]);
    *(__m128i *)(Y + 176) = _mm_add_epi16(*(__m128i *)&Y[176], *(__m128i *)&X[176]);
    *(__m128i *)(Y + 184) = _mm_add_epi16(*(__m128i *)&Y[184], *(__m128i *)&X[184]);
    *(__m128i *)(Y + 192) = _mm_add_epi16(*(__m128i *)&Y[192], *(__m128i *)&X[192]);
    *(__m128i *)(Y + 200) = _mm_add_epi16(*(__m128i *)&Y[200], *(__m128i *)&X[200]);
    *(__m128i *)(Y + 208) = _mm_add_epi16(*(__m128i *)&Y[208], *(__m128i *)&X[208]);
    *(__m128i *)(Y + 216) = _mm_add_epi16(*(__m128i *)&Y[216], *(__m128i *)&X[216]);
    *(__m128i *)(Y + 224) = _mm_add_epi16(*(__m128i *)&Y[224], *(__m128i *)&X[224]);
    *(__m128i *)(Y + 232) = _mm_add_epi16(*(__m128i *)&Y[232], *(__m128i *)&X[232]);
    *(__m128i *)(Y + 240) = _mm_add_epi16(*(__m128i *)&Y[240], *(__m128i *)&X[240]);
    *(__m128i *)(Y + 248) = _mm_add_epi16(*(__m128i *)&Y[248], *(__m128i *)&X[248]);
}

void HistgramSubShort(unsigned short *X, unsigned short *Y)
{
    *(__m128i *)(Y + 0) = _mm_sub_epi16(*(__m128i *)&Y[0], *(__m128i *)&X[0]);
    *(__m128i *)(Y + 8) = _mm_sub_epi16(*(__m128i *)&Y[8], *(__m128i *)&X[8]);
    *(__m128i *)(Y + 16) = _mm_sub_epi16(*(__m128i *)&Y[16], *(__m128i *)&X[16]);
    *(__m128i *)(Y + 24) = _mm_sub_epi16(*(__m128i *)&Y[24], *(__m128i *)&X[24]);
    *(__m128i *)(Y + 32) = _mm_sub_epi16(*(__m128i *)&Y[32], *(__m128i *)&X[32]);
    *(__m128i *)(Y + 40) = _mm_sub_epi16(*(__m128i *)&Y[40], *(__m128i *)&X[40]);
    *(__m128i *)(Y + 48) = _mm_sub_epi16(*(__m128i *)&Y[48], *(__m128i *)&X[48]);
    *(__m128i *)(Y + 56) = _mm_sub_epi16(*(__m128i *)&Y[56], *(__m128i *)&X[56]);
    *(__m128i *)(Y + 64) = _mm_sub_epi16(*(__m128i *)&Y[64], *(__m128i *)&X[64]);
    *(__m128i *)(Y + 72) = _mm_sub_epi16(*(__m128i *)&Y[72], *(__m128i *)&X[72]);
    *(__m128i *)(Y + 80) = _mm_sub_epi16(*(__m128i *)&Y[80], *(__m128i *)&X[80]);
    *(__m128i *)(Y + 88) = _mm_sub_epi16(*(__m128i *)&Y[88], *(__m128i *)&X[88]);
    *(__m128i *)(Y + 96) = _mm_sub_epi16(*(__m128i *)&Y[96], *(__m128i *)&X[96]);
    *(__m128i *)(Y + 104) = _mm_sub_epi16(*(__m128i *)&Y[104], *(__m128i *)&X[104]);
    *(__m128i *)(Y + 112) = _mm_sub_epi16(*(__m128i *)&Y[112], *(__m128i *)&X[112]);
    *(__m128i *)(Y + 120) = _mm_sub_epi16(*(__m128i *)&Y[120], *(__m128i *)&X[120]);
    *(__m128i *)(Y + 128) = _mm_sub_epi16(*(__m128i *)&Y[128], *(__m128i *)&X[128]);
    *(__m128i *)(Y + 136) = _mm_sub_epi16(*(__m128i *)&Y[136], *(__m128i *)&X[136]);
    *(__m128i *)(Y + 144) = _mm_sub_epi16(*(__m128i *)&Y[144], *(__m128i *)&X[144]);
    *(__m128i *)(Y + 152) = _mm_sub_epi16(*(__m128i *)&Y[152], *(__m128i *)&X[152]);
    *(__m128i *)(Y + 160) = _mm_sub_epi16(*(__m128i *)&Y[160], *(__m128i *)&X[160]);
    *(__m128i *)(Y + 168) = _mm_sub_epi16(*(__m128i *)&Y[168], *(__m128i *)&X[168]);
    *(__m128i *)(Y + 176) = _mm_sub_epi16(*(__m128i *)&Y[176], *(__m128i *)&X[176]);
    *(__m128i *)(Y + 184) = _mm_sub_epi16(*(__m128i *)&Y[184], *(__m128i *)&X[184]);
    *(__m128i *)(Y + 192) = _mm_sub_epi16(*(__m128i *)&Y[192], *(__m128i *)&X[192]);
    *(__m128i *)(Y + 200) = _mm_sub_epi16(*(__m128i *)&Y[200], *(__m128i *)&X[200]);
    *(__m128i *)(Y + 208) = _mm_sub_epi16(*(__m128i *)&Y[208], *(__m128i *)&X[208]);
    *(__m128i *)(Y + 216) = _mm_sub_epi16(*(__m128i *)&Y[216], *(__m128i *)&X[216]);
    *(__m128i *)(Y + 224) = _mm_sub_epi16(*(__m128i *)&Y[224], *(__m128i *)&X[224]);
    *(__m128i *)(Y + 232) = _mm_sub_epi16(*(__m128i *)&Y[232], *(__m128i *)&X[232]);
    *(__m128i *)(Y + 240) = _mm_sub_epi16(*(__m128i *)&Y[240], *(__m128i *)&X[240]);
    *(__m128i *)(Y + 248) = _mm_sub_epi16(*(__m128i *)&Y[248], *(__m128i *)&X[248]);
}

void HistgramSubAddShort(unsigned short *X, unsigned short *Y, unsigned short *Z)
{
    *(__m128i *)(Z + 0) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[0], *(__m128i *)&Z[0]), *(__m128i *)&X[0]);
    *(__m128i *)(Z + 8) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[8], *(__m128i *)&Z[8]), *(__m128i *)&X[8]);
    *(__m128i *)(Z + 16) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[16], *(__m128i *)&Z[16]), *(__m128i *)&X[16]);
    *(__m128i *)(Z + 24) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[24], *(__m128i *)&Z[24]), *(__m128i *)&X[24]);
    *(__m128i *)(Z + 32) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[32], *(__m128i *)&Z[32]), *(__m128i *)&X[32]);
    *(__m128i *)(Z + 40) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[40], *(__m128i *)&Z[40]), *(__m128i *)&X[40]);
    *(__m128i *)(Z + 48) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[48], *(__m128i *)&Z[48]), *(__m128i *)&X[48]);
    *(__m128i *)(Z + 56) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[56], *(__m128i *)&Z[56]), *(__m128i *)&X[56]);
    *(__m128i *)(Z + 64) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[64], *(__m128i *)&Z[64]), *(__m128i *)&X[64]);
    *(__m128i *)(Z + 72) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[72], *(__m128i *)&Z[72]), *(__m128i *)&X[72]);
    *(__m128i *)(Z + 80) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[80], *(__m128i *)&Z[80]), *(__m128i *)&X[80]);
    *(__m128i *)(Z + 88) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[88], *(__m128i *)&Z[88]), *(__m128i *)&X[88]);
    *(__m128i *)(Z + 96) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[96], *(__m128i *)&Z[96]), *(__m128i *)&X[96]);
    *(__m128i *)(Z + 104) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[104], *(__m128i *)&Z[104]), *(__m128i *)&X[104]);
    *(__m128i *)(Z + 112) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[112], *(__m128i *)&Z[112]), *(__m128i *)&X[112]);
    *(__m128i *)(Z + 120) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[120], *(__m128i *)&Z[120]), *(__m128i *)&X[120]);
    *(__m128i *)(Z + 128) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[128], *(__m128i *)&Z[128]), *(__m128i *)&X[128]);
    *(__m128i *)(Z + 136) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[136], *(__m128i *)&Z[136]), *(__m128i *)&X[136]);
    *(__m128i *)(Z + 144) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[144], *(__m128i *)&Z[144]), *(__m128i *)&X[144]);
    *(__m128i *)(Z + 152) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[152], *(__m128i *)&Z[152]), *(__m128i *)&X[152]);
    *(__m128i *)(Z + 160) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[160], *(__m128i *)&Z[160]), *(__m128i *)&X[160]);
    *(__m128i *)(Z + 168) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[168], *(__m128i *)&Z[168]), *(__m128i *)&X[168]);
    *(__m128i *)(Z + 176) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[176], *(__m128i *)&Z[176]), *(__m128i *)&X[176]);
    *(__m128i *)(Z + 184) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[184], *(__m128i *)&Z[184]), *(__m128i *)&X[184]);
    *(__m128i *)(Z + 192) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[192], *(__m128i *)&Z[192]), *(__m128i *)&X[192]);
    *(__m128i *)(Z + 200) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[200], *(__m128i *)&Z[200]), *(__m128i *)&X[200]);
    *(__m128i *)(Z + 208) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[208], *(__m128i *)&Z[208]), *(__m128i *)&X[208]);
    *(__m128i *)(Z + 216) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[216], *(__m128i *)&Z[216]), *(__m128i *)&X[216]);
    *(__m128i *)(Z + 224) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[224], *(__m128i *)&Z[224]), *(__m128i *)&X[224]);
    *(__m128i *)(Z + 232) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[232], *(__m128i *)&Z[232]), *(__m128i *)&X[232]);
    *(__m128i *)(Z + 240) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[240], *(__m128i *)&Z[240]), *(__m128i *)&X[240]);
    *(__m128i *)(Z + 248) = _mm_sub_epi16(_mm_add_epi16(*(__m128i *)&Y[248], *(__m128i *)&Z[248]), *(__m128i *)&X[248]);
}

// The most primitive algorithm
unsigned char Calc1(unsigned short *Hist, unsigned char Value, int Threshold)
{
    int Weight, Sum = 0, Divisor = 0;
    for (int Y = 0; Y < 256; Y++)
    {
        Weight = Hist[Y] * (2500 - abs(Y - Value) * 1000 / Threshold);
        if (Weight < 0)
            Weight = 0;
        Sum += Weight * Y;
        Divisor += Weight;
    }
    if (Divisor > 0)
        return (Sum + (Divisor >> 1)) / Divisor; // rounding
    else
        return Value;
}

// Improved algorithm
unsigned char Calc2(unsigned short *Hist, unsigned char Value, unsigned short *Intensity)
{
    int Weight = 0, Sum = 0, Divisor = 0;
    unsigned short *Offset = Intensity + 255 - Value;
    for (int Y = 0; Y < 256; Y++)
    {
        Weight = Hist[Y] * Offset[Y];
        Sum += Weight * Y;
        Divisor += Weight;
    }
    if (Divisor > 0)
        return (Sum + (Divisor >> 1)) / Divisor; // rounding
    else
        return Value;
}

// Improved algorithm
unsigned char Calc3(unsigned short *Hist, unsigned char Value, unsigned short *Intensity) //	The speed can be doubled after the cycle unfolds, because the multiplier inside is automatically optimized.
{
    int Weight = 0, Sum = 0, Divisor = 0;
    unsigned short *Offset = Intensity + 255 - Value;
    Weight = Hist[0] * Offset[0];
    Sum += Weight * 0;
    Divisor += Weight; //	Can I use parallelism with the instruction set, not tested
    Weight = Hist[1] * Offset[1];
    Sum += Weight * 1;
    Divisor += Weight;
    Weight = Hist[2] * Offset[2];
    Sum += Weight * 2;
    Divisor += Weight;
    Weight = Hist[3] * Offset[3];
    Sum += Weight * 3;
    Divisor += Weight;
    Weight = Hist[4] * Offset[4];
    Sum += Weight * 4;
    Divisor += Weight;
    Weight = Hist[5] * Offset[5];
    Sum += Weight * 5;
    Divisor += Weight;
    Weight = Hist[6] * Offset[6];
    Sum += Weight * 6;
    Divisor += Weight;
    Weight = Hist[7] * Offset[7];
    Sum += Weight * 7;
    Divisor += Weight;
    Weight = Hist[8] * Offset[8];
    Sum += Weight * 8;
    Divisor += Weight;
    Weight = Hist[9] * Offset[9];
    Sum += Weight * 9;
    Divisor += Weight;
    Weight = Hist[10] * Offset[10];
    Sum += Weight * 10;
    Divisor += Weight;
    Weight = Hist[11] * Offset[11];
    Sum += Weight * 11;
    Divisor += Weight;
    Weight = Hist[12] * Offset[12];
    Sum += Weight * 12;
    Divisor += Weight;
    Weight = Hist[13] * Offset[13];
    Sum += Weight * 13;
    Divisor += Weight;
    Weight = Hist[14] * Offset[14];
    Sum += Weight * 14;
    Divisor += Weight;
    Weight = Hist[15] * Offset[15];
    Sum += Weight * 15;
    Divisor += Weight;
    Weight = Hist[16] * Offset[16];
    Sum += Weight * 16;
    Divisor += Weight;
    Weight = Hist[17] * Offset[17];
    Sum += Weight * 17;
    Divisor += Weight;
    Weight = Hist[18] * Offset[18];
    Sum += Weight * 18;
    Divisor += Weight;
    Weight = Hist[19] * Offset[19];
    Sum += Weight * 19;
    Divisor += Weight;
    Weight = Hist[20] * Offset[20];
    Sum += Weight * 20;
    Divisor += Weight;
    Weight = Hist[21] * Offset[21];
    Sum += Weight * 21;
    Divisor += Weight;
    Weight = Hist[22] * Offset[22];
    Sum += Weight * 22;
    Divisor += Weight;
    Weight = Hist[23] * Offset[23];
    Sum += Weight * 23;
    Divisor += Weight;
    Weight = Hist[24] * Offset[24];
    Sum += Weight * 24;
    Divisor += Weight;
    Weight = Hist[25] * Offset[25];
    Sum += Weight * 25;
    Divisor += Weight;
    Weight = Hist[26] * Offset[26];
    Sum += Weight * 26;
    Divisor += Weight;
    Weight = Hist[27] * Offset[27];
    Sum += Weight * 27;
    Divisor += Weight;
    Weight = Hist[28] * Offset[28];
    Sum += Weight * 28;
    Divisor += Weight;
    Weight = Hist[29] * Offset[29];
    Sum += Weight * 29;
    Divisor += Weight;
    Weight = Hist[30] * Offset[30];
    Sum += Weight * 30;
    Divisor += Weight;
    Weight = Hist[31] * Offset[31];
    Sum += Weight * 31;
    Divisor += Weight;
    Weight = Hist[32] * Offset[32];
    Sum += Weight * 32;
    Divisor += Weight;
    Weight = Hist[33] * Offset[33];
    Sum += Weight * 33;
    Divisor += Weight;
    Weight = Hist[34] * Offset[34];
    Sum += Weight * 34;
    Divisor += Weight;
    Weight = Hist[35] * Offset[35];
    Sum += Weight * 35;
    Divisor += Weight;
    Weight = Hist[36] * Offset[36];
    Sum += Weight * 36;
    Divisor += Weight;
    Weight = Hist[37] * Offset[37];
    Sum += Weight * 37;
    Divisor += Weight;
    Weight = Hist[38] * Offset[38];
    Sum += Weight * 38;
    Divisor += Weight;
    Weight = Hist[39] * Offset[39];
    Sum += Weight * 39;
    Divisor += Weight;
    Weight = Hist[40] * Offset[40];
    Sum += Weight * 40;
    Divisor += Weight;
    Weight = Hist[41] * Offset[41];
    Sum += Weight * 41;
    Divisor += Weight;
    Weight = Hist[42] * Offset[42];
    Sum += Weight * 42;
    Divisor += Weight;
    Weight = Hist[43] * Offset[43];
    Sum += Weight * 43;
    Divisor += Weight;
    Weight = Hist[44] * Offset[44];
    Sum += Weight * 44;
    Divisor += Weight;
    Weight = Hist[45] * Offset[45];
    Sum += Weight * 45;
    Divisor += Weight;
    Weight = Hist[46] * Offset[46];
    Sum += Weight * 46;
    Divisor += Weight;
    Weight = Hist[47] * Offset[47];
    Sum += Weight * 47;
    Divisor += Weight;
    Weight = Hist[48] * Offset[48];
    Sum += Weight * 48;
    Divisor += Weight;
    Weight = Hist[49] * Offset[49];
    Sum += Weight * 49;
    Divisor += Weight;
    Weight = Hist[50] * Offset[50];
    Sum += Weight * 50;
    Divisor += Weight;
    Weight = Hist[51] * Offset[51];
    Sum += Weight * 51;
    Divisor += Weight;
    Weight = Hist[52] * Offset[52];
    Sum += Weight * 52;
    Divisor += Weight;
    Weight = Hist[53] * Offset[53];
    Sum += Weight * 53;
    Divisor += Weight;
    Weight = Hist[54] * Offset[54];
    Sum += Weight * 54;
    Divisor += Weight;
    Weight = Hist[55] * Offset[55];
    Sum += Weight * 55;
    Divisor += Weight;
    Weight = Hist[56] * Offset[56];
    Sum += Weight * 56;
    Divisor += Weight;
    Weight = Hist[57] * Offset[57];
    Sum += Weight * 57;
    Divisor += Weight;
    Weight = Hist[58] * Offset[58];
    Sum += Weight * 58;
    Divisor += Weight;
    Weight = Hist[59] * Offset[59];
    Sum += Weight * 59;
    Divisor += Weight;
    Weight = Hist[60] * Offset[60];
    Sum += Weight * 60;
    Divisor += Weight;
    Weight = Hist[61] * Offset[61];
    Sum += Weight * 61;
    Divisor += Weight;
    Weight = Hist[62] * Offset[62];
    Sum += Weight * 62;
    Divisor += Weight;
    Weight = Hist[63] * Offset[63];
    Sum += Weight * 63;
    Divisor += Weight;
    Weight = Hist[64] * Offset[64];
    Sum += Weight * 64;
    Divisor += Weight;
    Weight = Hist[65] * Offset[65];
    Sum += Weight * 65;
    Divisor += Weight;
    Weight = Hist[66] * Offset[66];
    Sum += Weight * 66;
    Divisor += Weight;
    Weight = Hist[67] * Offset[67];
    Sum += Weight * 67;
    Divisor += Weight;
    Weight = Hist[68] * Offset[68];
    Sum += Weight * 68;
    Divisor += Weight;
    Weight = Hist[69] * Offset[69];
    Sum += Weight * 69;
    Divisor += Weight;
    Weight = Hist[70] * Offset[70];
    Sum += Weight * 70;
    Divisor += Weight;
    Weight = Hist[71] * Offset[71];
    Sum += Weight * 71;
    Divisor += Weight;
    Weight = Hist[72] * Offset[72];
    Sum += Weight * 72;
    Divisor += Weight;
    Weight = Hist[73] * Offset[73];
    Sum += Weight * 73;
    Divisor += Weight;
    Weight = Hist[74] * Offset[74];
    Sum += Weight * 74;
    Divisor += Weight;
    Weight = Hist[75] * Offset[75];
    Sum += Weight * 75;
    Divisor += Weight;
    Weight = Hist[76] * Offset[76];
    Sum += Weight * 76;
    Divisor += Weight;
    Weight = Hist[77] * Offset[77];
    Sum += Weight * 77;
    Divisor += Weight;
    Weight = Hist[78] * Offset[78];
    Sum += Weight * 78;
    Divisor += Weight;
    Weight = Hist[79] * Offset[79];
    Sum += Weight * 79;
    Divisor += Weight;
    Weight = Hist[80] * Offset[80];
    Sum += Weight * 80;
    Divisor += Weight;
    Weight = Hist[81] * Offset[81];
    Sum += Weight * 81;
    Divisor += Weight;
    Weight = Hist[82] * Offset[82];
    Sum += Weight * 82;
    Divisor += Weight;
    Weight = Hist[83] * Offset[83];
    Sum += Weight * 83;
    Divisor += Weight;
    Weight = Hist[84] * Offset[84];
    Sum += Weight * 84;
    Divisor += Weight;
    Weight = Hist[85] * Offset[85];
    Sum += Weight * 85;
    Divisor += Weight;
    Weight = Hist[86] * Offset[86];
    Sum += Weight * 86;
    Divisor += Weight;
    Weight = Hist[87] * Offset[87];
    Sum += Weight * 87;
    Divisor += Weight;
    Weight = Hist[88] * Offset[88];
    Sum += Weight * 88;
    Divisor += Weight;
    Weight = Hist[89] * Offset[89];
    Sum += Weight * 89;
    Divisor += Weight;
    Weight = Hist[90] * Offset[90];
    Sum += Weight * 90;
    Divisor += Weight;
    Weight = Hist[91] * Offset[91];
    Sum += Weight * 91;
    Divisor += Weight;
    Weight = Hist[92] * Offset[92];
    Sum += Weight * 92;
    Divisor += Weight;
    Weight = Hist[93] * Offset[93];
    Sum += Weight * 93;
    Divisor += Weight;
    Weight = Hist[94] * Offset[94];
    Sum += Weight * 94;
    Divisor += Weight;
    Weight = Hist[95] * Offset[95];
    Sum += Weight * 95;
    Divisor += Weight;
    Weight = Hist[96] * Offset[96];
    Sum += Weight * 96;
    Divisor += Weight;
    Weight = Hist[97] * Offset[97];
    Sum += Weight * 97;
    Divisor += Weight;
    Weight = Hist[98] * Offset[98];
    Sum += Weight * 98;
    Divisor += Weight;
    Weight = Hist[99] * Offset[99];
    Sum += Weight * 99;
    Divisor += Weight;
    Weight = Hist[100] * Offset[100];
    Sum += Weight * 100;
    Divisor += Weight;
    Weight = Hist[101] * Offset[101];
    Sum += Weight * 101;
    Divisor += Weight;
    Weight = Hist[102] * Offset[102];
    Sum += Weight * 102;
    Divisor += Weight;
    Weight = Hist[103] * Offset[103];
    Sum += Weight * 103;
    Divisor += Weight;
    Weight = Hist[104] * Offset[104];
    Sum += Weight * 104;
    Divisor += Weight;
    Weight = Hist[105] * Offset[105];
    Sum += Weight * 105;
    Divisor += Weight;
    Weight = Hist[106] * Offset[106];
    Sum += Weight * 106;
    Divisor += Weight;
    Weight = Hist[107] * Offset[107];
    Sum += Weight * 107;
    Divisor += Weight;
    Weight = Hist[108] * Offset[108];
    Sum += Weight * 108;
    Divisor += Weight;
    Weight = Hist[109] * Offset[109];
    Sum += Weight * 109;
    Divisor += Weight;
    Weight = Hist[110] * Offset[110];
    Sum += Weight * 110;
    Divisor += Weight;
    Weight = Hist[111] * Offset[111];
    Sum += Weight * 111;
    Divisor += Weight;
    Weight = Hist[112] * Offset[112];
    Sum += Weight * 112;
    Divisor += Weight;
    Weight = Hist[113] * Offset[113];
    Sum += Weight * 113;
    Divisor += Weight;
    Weight = Hist[114] * Offset[114];
    Sum += Weight * 114;
    Divisor += Weight;
    Weight = Hist[115] * Offset[115];
    Sum += Weight * 115;
    Divisor += Weight;
    Weight = Hist[116] * Offset[116];
    Sum += Weight * 116;
    Divisor += Weight;
    Weight = Hist[117] * Offset[117];
    Sum += Weight * 117;
    Divisor += Weight;
    Weight = Hist[118] * Offset[118];
    Sum += Weight * 118;
    Divisor += Weight;
    Weight = Hist[119] * Offset[119];
    Sum += Weight * 119;
    Divisor += Weight;
    Weight = Hist[120] * Offset[120];
    Sum += Weight * 120;
    Divisor += Weight;
    Weight = Hist[121] * Offset[121];
    Sum += Weight * 121;
    Divisor += Weight;
    Weight = Hist[122] * Offset[122];
    Sum += Weight * 122;
    Divisor += Weight;
    Weight = Hist[123] * Offset[123];
    Sum += Weight * 123;
    Divisor += Weight;
    Weight = Hist[124] * Offset[124];
    Sum += Weight * 124;
    Divisor += Weight;
    Weight = Hist[125] * Offset[125];
    Sum += Weight * 125;
    Divisor += Weight;
    Weight = Hist[126] * Offset[126];
    Sum += Weight * 126;
    Divisor += Weight;
    Weight = Hist[127] * Offset[127];
    Sum += Weight * 127;
    Divisor += Weight;
    Weight = Hist[128] * Offset[128];
    Sum += Weight * 128;
    Divisor += Weight;
    Weight = Hist[129] * Offset[129];
    Sum += Weight * 129;
    Divisor += Weight;
    Weight = Hist[130] * Offset[130];
    Sum += Weight * 130;
    Divisor += Weight;
    Weight = Hist[131] * Offset[131];
    Sum += Weight * 131;
    Divisor += Weight;
    Weight = Hist[132] * Offset[132];
    Sum += Weight * 132;
    Divisor += Weight;
    Weight = Hist[133] * Offset[133];
    Sum += Weight * 133;
    Divisor += Weight;
    Weight = Hist[134] * Offset[134];
    Sum += Weight * 134;
    Divisor += Weight;
    Weight = Hist[135] * Offset[135];
    Sum += Weight * 135;
    Divisor += Weight;
    Weight = Hist[136] * Offset[136];
    Sum += Weight * 136;
    Divisor += Weight;
    Weight = Hist[137] * Offset[137];
    Sum += Weight * 137;
    Divisor += Weight;
    Weight = Hist[138] * Offset[138];
    Sum += Weight * 138;
    Divisor += Weight;
    Weight = Hist[139] * Offset[139];
    Sum += Weight * 139;
    Divisor += Weight;
    Weight = Hist[140] * Offset[140];
    Sum += Weight * 140;
    Divisor += Weight;
    Weight = Hist[141] * Offset[141];
    Sum += Weight * 141;
    Divisor += Weight;
    Weight = Hist[142] * Offset[142];
    Sum += Weight * 142;
    Divisor += Weight;
    Weight = Hist[143] * Offset[143];
    Sum += Weight * 143;
    Divisor += Weight;
    Weight = Hist[144] * Offset[144];
    Sum += Weight * 144;
    Divisor += Weight;
    Weight = Hist[145] * Offset[145];
    Sum += Weight * 145;
    Divisor += Weight;
    Weight = Hist[146] * Offset[146];
    Sum += Weight * 146;
    Divisor += Weight;
    Weight = Hist[147] * Offset[147];
    Sum += Weight * 147;
    Divisor += Weight;
    Weight = Hist[148] * Offset[148];
    Sum += Weight * 148;
    Divisor += Weight;
    Weight = Hist[149] * Offset[149];
    Sum += Weight * 149;
    Divisor += Weight;
    Weight = Hist[150] * Offset[150];
    Sum += Weight * 150;
    Divisor += Weight;
    Weight = Hist[151] * Offset[151];
    Sum += Weight * 151;
    Divisor += Weight;
    Weight = Hist[152] * Offset[152];
    Sum += Weight * 152;
    Divisor += Weight;
    Weight = Hist[153] * Offset[153];
    Sum += Weight * 153;
    Divisor += Weight;
    Weight = Hist[154] * Offset[154];
    Sum += Weight * 154;
    Divisor += Weight;
    Weight = Hist[155] * Offset[155];
    Sum += Weight * 155;
    Divisor += Weight;
    Weight = Hist[156] * Offset[156];
    Sum += Weight * 156;
    Divisor += Weight;
    Weight = Hist[157] * Offset[157];
    Sum += Weight * 157;
    Divisor += Weight;
    Weight = Hist[158] * Offset[158];
    Sum += Weight * 158;
    Divisor += Weight;
    Weight = Hist[159] * Offset[159];
    Sum += Weight * 159;
    Divisor += Weight;
    Weight = Hist[160] * Offset[160];
    Sum += Weight * 160;
    Divisor += Weight;
    Weight = Hist[161] * Offset[161];
    Sum += Weight * 161;
    Divisor += Weight;
    Weight = Hist[162] * Offset[162];
    Sum += Weight * 162;
    Divisor += Weight;
    Weight = Hist[163] * Offset[163];
    Sum += Weight * 163;
    Divisor += Weight;
    Weight = Hist[164] * Offset[164];
    Sum += Weight * 164;
    Divisor += Weight;
    Weight = Hist[165] * Offset[165];
    Sum += Weight * 165;
    Divisor += Weight;
    Weight = Hist[166] * Offset[166];
    Sum += Weight * 166;
    Divisor += Weight;
    Weight = Hist[167] * Offset[167];
    Sum += Weight * 167;
    Divisor += Weight;
    Weight = Hist[168] * Offset[168];
    Sum += Weight * 168;
    Divisor += Weight;
    Weight = Hist[169] * Offset[169];
    Sum += Weight * 169;
    Divisor += Weight;
    Weight = Hist[170] * Offset[170];
    Sum += Weight * 170;
    Divisor += Weight;
    Weight = Hist[171] * Offset[171];
    Sum += Weight * 171;
    Divisor += Weight;
    Weight = Hist[172] * Offset[172];
    Sum += Weight * 172;
    Divisor += Weight;
    Weight = Hist[173] * Offset[173];
    Sum += Weight * 173;
    Divisor += Weight;
    Weight = Hist[174] * Offset[174];
    Sum += Weight * 174;
    Divisor += Weight;
    Weight = Hist[175] * Offset[175];
    Sum += Weight * 175;
    Divisor += Weight;
    Weight = Hist[176] * Offset[176];
    Sum += Weight * 176;
    Divisor += Weight;
    Weight = Hist[177] * Offset[177];
    Sum += Weight * 177;
    Divisor += Weight;
    Weight = Hist[178] * Offset[178];
    Sum += Weight * 178;
    Divisor += Weight;
    Weight = Hist[179] * Offset[179];
    Sum += Weight * 179;
    Divisor += Weight;
    Weight = Hist[180] * Offset[180];
    Sum += Weight * 180;
    Divisor += Weight;
    Weight = Hist[181] * Offset[181];
    Sum += Weight * 181;
    Divisor += Weight;
    Weight = Hist[182] * Offset[182];
    Sum += Weight * 182;
    Divisor += Weight;
    Weight = Hist[183] * Offset[183];
    Sum += Weight * 183;
    Divisor += Weight;
    Weight = Hist[184] * Offset[184];
    Sum += Weight * 184;
    Divisor += Weight;
    Weight = Hist[185] * Offset[185];
    Sum += Weight * 185;
    Divisor += Weight;
    Weight = Hist[186] * Offset[186];
    Sum += Weight * 186;
    Divisor += Weight;
    Weight = Hist[187] * Offset[187];
    Sum += Weight * 187;
    Divisor += Weight;
    Weight = Hist[188] * Offset[188];
    Sum += Weight * 188;
    Divisor += Weight;
    Weight = Hist[189] * Offset[189];
    Sum += Weight * 189;
    Divisor += Weight;
    Weight = Hist[190] * Offset[190];
    Sum += Weight * 190;
    Divisor += Weight;
    Weight = Hist[191] * Offset[191];
    Sum += Weight * 191;
    Divisor += Weight;
    Weight = Hist[192] * Offset[192];
    Sum += Weight * 192;
    Divisor += Weight;
    Weight = Hist[193] * Offset[193];
    Sum += Weight * 193;
    Divisor += Weight;
    Weight = Hist[194] * Offset[194];
    Sum += Weight * 194;
    Divisor += Weight;
    Weight = Hist[195] * Offset[195];
    Sum += Weight * 195;
    Divisor += Weight;
    Weight = Hist[196] * Offset[196];
    Sum += Weight * 196;
    Divisor += Weight;
    Weight = Hist[197] * Offset[197];
    Sum += Weight * 197;
    Divisor += Weight;
    Weight = Hist[198] * Offset[198];
    Sum += Weight * 198;
    Divisor += Weight;
    Weight = Hist[199] * Offset[199];
    Sum += Weight * 199;
    Divisor += Weight;
    Weight = Hist[200] * Offset[200];
    Sum += Weight * 200;
    Divisor += Weight;
    Weight = Hist[201] * Offset[201];
    Sum += Weight * 201;
    Divisor += Weight;
    Weight = Hist[202] * Offset[202];
    Sum += Weight * 202;
    Divisor += Weight;
    Weight = Hist[203] * Offset[203];
    Sum += Weight * 203;
    Divisor += Weight;
    Weight = Hist[204] * Offset[204];
    Sum += Weight * 204;
    Divisor += Weight;
    Weight = Hist[205] * Offset[205];
    Sum += Weight * 205;
    Divisor += Weight;
    Weight = Hist[206] * Offset[206];
    Sum += Weight * 206;
    Divisor += Weight;
    Weight = Hist[207] * Offset[207];
    Sum += Weight * 207;
    Divisor += Weight;
    Weight = Hist[208] * Offset[208];
    Sum += Weight * 208;
    Divisor += Weight;
    Weight = Hist[209] * Offset[209];
    Sum += Weight * 209;
    Divisor += Weight;
    Weight = Hist[210] * Offset[210];
    Sum += Weight * 210;
    Divisor += Weight;
    Weight = Hist[211] * Offset[211];
    Sum += Weight * 211;
    Divisor += Weight;
    Weight = Hist[212] * Offset[212];
    Sum += Weight * 212;
    Divisor += Weight;
    Weight = Hist[213] * Offset[213];
    Sum += Weight * 213;
    Divisor += Weight;
    Weight = Hist[214] * Offset[214];
    Sum += Weight * 214;
    Divisor += Weight;
    Weight = Hist[215] * Offset[215];
    Sum += Weight * 215;
    Divisor += Weight;
    Weight = Hist[216] * Offset[216];
    Sum += Weight * 216;
    Divisor += Weight;
    Weight = Hist[217] * Offset[217];
    Sum += Weight * 217;
    Divisor += Weight;
    Weight = Hist[218] * Offset[218];
    Sum += Weight * 218;
    Divisor += Weight;
    Weight = Hist[219] * Offset[219];
    Sum += Weight * 219;
    Divisor += Weight;
    Weight = Hist[220] * Offset[220];
    Sum += Weight * 220;
    Divisor += Weight;
    Weight = Hist[221] * Offset[221];
    Sum += Weight * 221;
    Divisor += Weight;
    Weight = Hist[222] * Offset[222];
    Sum += Weight * 222;
    Divisor += Weight;
    Weight = Hist[223] * Offset[223];
    Sum += Weight * 223;
    Divisor += Weight;
    Weight = Hist[224] * Offset[224];
    Sum += Weight * 224;
    Divisor += Weight;
    Weight = Hist[225] * Offset[225];
    Sum += Weight * 225;
    Divisor += Weight;
    Weight = Hist[226] * Offset[226];
    Sum += Weight * 226;
    Divisor += Weight;
    Weight = Hist[227] * Offset[227];
    Sum += Weight * 227;
    Divisor += Weight;
    Weight = Hist[228] * Offset[228];
    Sum += Weight * 228;
    Divisor += Weight;
    Weight = Hist[229] * Offset[229];
    Sum += Weight * 229;
    Divisor += Weight;
    Weight = Hist[230] * Offset[230];
    Sum += Weight * 230;
    Divisor += Weight;
    Weight = Hist[231] * Offset[231];
    Sum += Weight * 231;
    Divisor += Weight;
    Weight = Hist[232] * Offset[232];
    Sum += Weight * 232;
    Divisor += Weight;
    Weight = Hist[233] * Offset[233];
    Sum += Weight * 233;
    Divisor += Weight;
    Weight = Hist[234] * Offset[234];
    Sum += Weight * 234;
    Divisor += Weight;
    Weight = Hist[235] * Offset[235];
    Sum += Weight * 235;
    Divisor += Weight;
    Weight = Hist[236] * Offset[236];
    Sum += Weight * 236;
    Divisor += Weight;
    Weight = Hist[237] * Offset[237];
    Sum += Weight * 237;
    Divisor += Weight;
    Weight = Hist[238] * Offset[238];
    Sum += Weight * 238;
    Divisor += Weight;
    Weight = Hist[239] * Offset[239];
    Sum += Weight * 239;
    Divisor += Weight;
    Weight = Hist[240] * Offset[240];
    Sum += Weight * 240;
    Divisor += Weight;
    Weight = Hist[241] * Offset[241];
    Sum += Weight * 241;
    Divisor += Weight;
    Weight = Hist[242] * Offset[242];
    Sum += Weight * 242;
    Divisor += Weight;
    Weight = Hist[243] * Offset[243];
    Sum += Weight * 243;
    Divisor += Weight;
    Weight = Hist[244] * Offset[244];
    Sum += Weight * 244;
    Divisor += Weight;
    Weight = Hist[245] * Offset[245];
    Sum += Weight * 245;
    Divisor += Weight;
    Weight = Hist[246] * Offset[246];
    Sum += Weight * 246;
    Divisor += Weight;
    Weight = Hist[247] * Offset[247];
    Sum += Weight * 247;
    Divisor += Weight;
    Weight = Hist[248] * Offset[248];
    Sum += Weight * 248;
    Divisor += Weight;
    Weight = Hist[249] * Offset[249];
    Sum += Weight * 249;
    Divisor += Weight;
    Weight = Hist[250] * Offset[250];
    Sum += Weight * 250;
    Divisor += Weight;
    Weight = Hist[251] * Offset[251];
    Sum += Weight * 251;
    Divisor += Weight;
    Weight = Hist[252] * Offset[252];
    Sum += Weight * 252;
    Divisor += Weight;
    Weight = Hist[253] * Offset[253];
    Sum += Weight * 253;
    Divisor += Weight;
    Weight = Hist[254] * Offset[254];
    Sum += Weight * 254;
    Divisor += Weight;
    Weight = Hist[255] * Offset[255];
    Sum += Weight * 255;
    Divisor += Weight;
    if (Divisor > 0)
        return (Sum + (Divisor >> 1)) / Divisor; //	rounding
    else
        return Value;
}

//	SSE-optimized algorithm
unsigned char Calc4(unsigned short *Hist, unsigned char Value, unsigned short *Intensity, unsigned short *Level)
{
    unsigned short *Offset = Intensity + 255 - Value;
    __m128i SumS = _mm_setzero_si128();
    __m128i WeightS = _mm_setzero_si128();
    for (int K = 0; K < 256; K += 8)
    {
        __m128i H = _mm_load_si128((__m128i const *)(Hist + K));
        __m128i L = _mm_load_si128((__m128i const *)(Level + K)); // Ability to use 256-bit AVX registers
        __m128i I = _mm_loadu_si128((__m128i const *)(Offset + K));
        SumS = _mm_add_epi32(_mm_madd_epi16(_mm_mullo_epi16(L, I), H), SumS);
        WeightS = _mm_add_epi32(_mm_madd_epi16(H, I), WeightS);
    }
    const int *WW = (const int *)&WeightS;
    const int *SS = (const int *)&SumS;

    int Sum = SS[0] + SS[1] + SS[2] + SS[3];
    int Divisor = WW[0] + WW[1] + WW[2] + WW[3];
    if (Divisor > 0)
        return (Sum + (Divisor >> 1)) / Divisor; //	rounding
    else
        return Value;
}

// Convert RGB data to single channel, this can be slightly faster
void SplitRGB(unsigned char *Src, unsigned char *Blue, unsigned char *Green, unsigned char *Red, int Width, int Height, int Stride)
{
    for (int Y = 0; Y < Height; Y++)
    {
        unsigned char *PointerS = Src + Y * Stride;
        unsigned char *PointerB = Blue + Y * Width;
        unsigned char *PointerG = Green + Y * Width;
        unsigned char *PointerR = Red + Y * Width;
        for (int X = 0; X < Width; X++)
        {
            PointerB[X] = PointerS[0];
            PointerG[X] = PointerS[1];
            PointerR[X] = PointerS[2];
            PointerS += 3;
        }
    }
}

//	Convert single-channel data to RGB data, which can be slightly faster
void CombineRGB(unsigned char *Blue, unsigned char *Green, unsigned char *Red, unsigned char *Dest, int Width, int Height, int Stride)
{
    for (int Y = 0; Y < Height; Y++)
    {
        unsigned char *PointerD = Dest + Y * Stride;
        unsigned char *PointerB = Blue + Y * Width;
        unsigned char *PointerG = Green + Y * Width;
        unsigned char *PointerR = Red + Y * Width;
        for (int X = 0; X < Width; X++)
        {
            PointerD[0] = PointerB[X];
            PointerD[1] = PointerG[X];
            PointerD[2] = PointerR[X];
            PointerD += 3;
        }
    }
}

void SurfaceBlur(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, int Radius, int Threshold)
{
    int Channel = Stride / Width;
    if ((Channel != 1) && (Channel != 3))
        return;

    if (Channel == 1)
    {
        unsigned short *ColHist = (unsigned short *)aligned_alloc(32, 256 * (Width + Radius + Radius) * sizeof(unsigned short)); // Considering the need of advanced functions such as SSE, AVX, the allocation start address uses 32-byte alignment. This function is actually _mm_malloc
        unsigned short *Hist = (unsigned short *)aligned_alloc(32, 256 * sizeof(unsigned short));

        unsigned short *Intensity = (unsigned short *)aligned_alloc(32, 511 * sizeof(unsigned short)); //	Avoid abs when a negative value is used
        unsigned short *Level = (unsigned short *)aligned_alloc(32, 256 * sizeof(unsigned short));

        int *RowOffset = (int *)malloc((Width + Radius + Radius) * sizeof(int));
        int *ColOffset = (int *)malloc((Height + Radius + Radius) * sizeof(int));

        GetOffsetPos(RowOffset, Width, Radius, Radius);
        GetOffsetPos(ColOffset, Height, Radius, Radius);

        memset(ColHist, 0, 256 * (Width + Radius + Radius) * sizeof(unsigned short)); //	Make sure to clear

        for (int Y = 0; Y < 256; Y++)
            Level[Y] = Y; //	This is used for CalcSSE

        //	In order not to overflow with SSE, the data here is reduced, of course, the accuracy of the algorithm is reduced
        for (int Y = -255; Y <= 255; Y++)
        {
            int Factor = (255 - abs(Y) * 100 / Threshold);
            if (Factor < 0)
                Factor = 0;
            Intensity[Y + 255] = Factor / 2;
        }

        for (int Y = 0; Y < Height; Y++)
        {
            if (Y == 0) //	The first row of column histograms
            {
                for (int K = -Radius; K <= Radius; K++)
                {
                    unsigned char *LinePS = Src + ColOffset[K + Radius] * Stride;
                    for (int X = -Radius; X < Width + Radius; X++)
                    {
                        ColHist[(X + Radius) * 256 + LinePS[RowOffset[X + Radius]]]++;
                    }
                }
            }
            else //	Column histogram for other rows, update it
            {
                unsigned char *LinePS = Src + ColOffset[Y - 1] * Stride;
                for (int X = -Radius; X < Width + Radius; X++) // Delete the histogram data for the row that is out of range
                {
                    ColHist[(X + Radius) * 256 + LinePS[RowOffset[X + Radius]]]--;
                }

                LinePS = Src + ColOffset[Y + Radius + Radius] * Stride;
                for (int X = -Radius; X < Width + Radius; X++) // Increase the histogram data for the line in the incoming range
                {
                    ColHist[(X + Radius) * 256 + LinePS[RowOffset[X + Radius]]]++;
                }
            }

            memset(Hist, 0, 256 * sizeof(unsigned short)); //	Each row of histogram data is cleared first

            unsigned char *LinePS = Src + Y * Stride;
            unsigned char *LinePD = Dest + Y * Stride;

            for (int X = 0; X < Width; X++)
            {
                if (X == 0)
                {
                    for (int K = -Radius; K <= Radius; K++) //	First pixel, needs to be recalculated
                        HistgramAddShort(ColHist + (K + Radius) * 256, Hist);
                }
                else
                {
                    //HistgramAddShort(ColHist + (RowOffset[X + Radius + Radius] + Radius) * 256, Hist);		//	Writing separately is slower than merge writing
                    //HistgramSubShort(ColHist + (RowOffset[X - 1] + Radius) * 256, Hist);
                    HistgramSubAddShort(ColHist + (RowOffset[X - 1] + Radius) * 256, ColHist + (RowOffset[X + Radius + Radius] + Radius) * 256, Hist); //	The other pixels in the line can be deleted and added in turn.
                }

                //LinePD[X] = Calc1(Hist, LinePS[X], Threshold);
                //LinePD[X] = Calc2(Hist, LinePS[X], Intensity);
                //LinePD[X] = Calc3(Hist, LinePS[X], Intensity);
                LinePD[X] = Calc4(Hist, LinePS[X], Intensity, Level);
            }
        }
        free(ColHist);
        free(Hist);
        free(Intensity);
        free(Level);
        free(RowOffset);
        free(ColOffset);
    }
    else
    {
        unsigned char *SrcB = (unsigned char *)malloc(Width * Height * sizeof(unsigned char));
        unsigned char *SrcG = (unsigned char *)malloc(Width * Height * sizeof(unsigned char));
        unsigned char *SrcR = (unsigned char *)malloc(Width * Height * sizeof(unsigned char));

        unsigned char *DstB = (unsigned char *)malloc(Width * Height * sizeof(unsigned char));
        unsigned char *DstG = (unsigned char *)malloc(Width * Height * sizeof(unsigned char));
        unsigned char *DstR = (unsigned char *)malloc(Width * Height * sizeof(unsigned char));

        SplitRGB(Src, SrcB, SrcG, SrcR, Width, Height, Stride);
#pragma omp parallel sections num_threads(3)
        {
#pragma omp section
            SurfaceBlur(SrcB, DstB, Width, Height, Width, Radius, Threshold);
#pragma omp section
            SurfaceBlur(SrcG, DstG, Width, Height, Width, Radius, Threshold);
#pragma omp section
            SurfaceBlur(SrcR, DstR, Width, Height, Width, Radius, Threshold);
        }
        CombineRGB(DstB, DstG, DstR, Dest, Width, Height, Stride);

        free(SrcB);
        free(SrcG);
        free(SrcR);
        free(DstB);
        free(DstG);
        free(DstR);
    }
}

int main()
{
    char *img_path = "../0.jpg";
    cv::Mat img = cv::imread(img_path);
    cv::Mat dst(img.rows, img.cols, img.type());
    SurfaceBlur(img.data, dst.data, img.cols, img.rows, img.cols * img.channels(), 10, 15);
    cv::imwrite("0_r3.jpg", dst);
}